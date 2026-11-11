//! Narrow GGUF v2/v3 reader: mmap the file, parse header, metadata and tensor-info table.
//!
//! A port of the built-in parser in `tools/model_inspect.py` (lab repository)
//! (`parse_builtin`), including its handling of long arrays (> 4096 items are counted,
//! not stored) and the data-section alignment rule. Tensor payloads are never read here.

use memmap2::Mmap;
use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Arrays longer than this are kept as a length only (same bound as model_inspect.py).
pub const LONG_ARRAY: u64 = 4096;
/// GGML_MAX_DIMS
pub const MAX_DIMS: u32 = 4;

#[derive(Debug)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

/// ggml type ids (ggml.h at the pinned commit) with block size and block bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GgmlType(pub u32);

impl GgmlType {
    pub const F32: GgmlType = GgmlType(0);
    pub const F16: GgmlType = GgmlType(1);
    pub const Q8_0: GgmlType = GgmlType(8);
    pub const Q3_K: GgmlType = GgmlType(11);
    pub const Q4_K: GgmlType = GgmlType(12);
    pub const Q5_K: GgmlType = GgmlType(13);
    pub const Q6_K: GgmlType = GgmlType(14);

    /// (name, block elements, block bytes); same table as model_inspect.py GGML_TYPES.
    pub fn info(self) -> Option<(&'static str, u64, u64)> {
        Some(match self.0 {
            0 => ("F32", 1, 4),
            1 => ("F16", 1, 2),
            2 => ("Q4_0", 32, 18),
            3 => ("Q4_1", 32, 20),
            6 => ("Q5_0", 32, 22),
            7 => ("Q5_1", 32, 24),
            8 => ("Q8_0", 32, 34),
            9 => ("Q8_1", 32, 36),
            10 => ("Q2_K", 256, 84),
            11 => ("Q3_K", 256, 110),
            12 => ("Q4_K", 256, 144),
            13 => ("Q5_K", 256, 176),
            14 => ("Q6_K", 256, 210),
            15 => ("Q8_K", 256, 292),
            16 => ("IQ2_XXS", 256, 66),
            17 => ("IQ2_XS", 256, 74),
            18 => ("IQ3_XXS", 256, 98),
            19 => ("IQ1_S", 256, 50),
            20 => ("IQ4_NL", 32, 18),
            21 => ("IQ3_S", 256, 110),
            22 => ("IQ2_S", 256, 82),
            23 => ("IQ4_XS", 256, 136),
            24 => ("I8", 1, 1),
            25 => ("I16", 1, 2),
            26 => ("I32", 1, 4),
            27 => ("I64", 1, 8),
            28 => ("F64", 1, 8),
            29 => ("IQ1_M", 256, 56),
            30 => ("BF16", 1, 2),
            34 => ("TQ1_0", 256, 54),
            35 => ("TQ2_0", 256, 66),
            39 => ("MXFP4", 32, 17),
            40 => ("NVFP4", 64, 36),
            41 => ("Q1_0", 128, 18),
            42 => ("Q2_0", 64, 18),
            43 => ("F8_E4M3", 1, 1),
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        self.info().map(|i| i.0).unwrap_or("UNKNOWN")
    }
    pub fn from_name(name: &str) -> Option<GgmlType> {
        (0..64).map(GgmlType).find(|t| t.info().map(|i| i.0) == Some(name))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    /// An array longer than [`LONG_ARRAY`]: only its item type and length are kept.
    LongArray { item_type: u32, len: u64 },
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::I64(v) if v >= 0 => v as u64,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        Some(match *self {
            Value::F32(v) => v as f64,
            Value::F64(v) => v,
            _ => return self.as_u64().map(|v| v as f64),
        })
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: shape[0] is the contiguous (row) dimension.
    pub shape: Vec<u64>,
    pub ty: GgmlType,
    /// Offset relative to the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Element count, `None` on u64 overflow (a crafted header).
    pub fn checked_elements(&self) -> Option<u64> {
        self.shape.iter().try_fold(1u64, |a, &d| a.checked_mul(d))
    }
    /// Data size in bytes, `None` on overflow or an unknown type.
    pub fn checked_bytes(&self) -> Option<u64> {
        let (_, blk, size) = self.ty.info()?;
        (self.checked_elements()? / blk).checked_mul(size)
    }
    /// [`parse_header`] guarantees these do not overflow; a hand-built header that does gets
    /// `u64::MAX`, which fails every bounds check (REVIEW-v0.1.1 S1).
    pub fn elements(&self) -> u64 {
        self.checked_elements().unwrap_or(u64::MAX)
    }
    pub fn bytes(&self) -> u64 {
        self.checked_bytes().unwrap_or(u64::MAX)
    }
}

/// Parsed header; independent of the mmap so it can also be built from fixtures.
#[derive(Clone, Debug, Default)]
pub struct Header {
    pub version: u32,
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    pub header_bytes_read: u64,
}

impl Header {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture").and_then(|v| v.as_str())
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos.checked_add(n).map_or(true, |e| e > self.buf.len()) {
            return err("truncated GGUF header");
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

fn scalar_size(vtype: u32) -> Option<usize> {
    Some(match vtype {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4 | 5 | 6 => 4,
        10 | 11 | 12 => 8,
        _ => return None,
    })
}

fn value(c: &mut Cursor, vtype: u32) -> Result<Value> {
    fn b<'a>(c: &mut Cursor<'a>, n: usize) -> Result<&'a [u8]> {
        c.take(n)
    }
    Ok(match vtype {
        0 => Value::U8(b(c, 1)?[0]),
        1 => Value::I8(b(c, 1)?[0] as i8),
        2 => Value::U16(u16::from_le_bytes(b(c, 2)?.try_into().unwrap())),
        3 => Value::I16(i16::from_le_bytes(b(c, 2)?.try_into().unwrap())),
        4 => Value::U32(c.u32()?),
        5 => Value::I32(c.u32()? as i32),
        6 => Value::F32(f32::from_le_bytes(b(c, 4)?.try_into().unwrap())),
        7 => Value::Bool(b(c, 1)?[0] != 0),
        8 => Value::Str(c.string()?),
        9 => {
            let item = c.u32()?;
            let n = c.u64()?;
            if n > LONG_ARRAY {
                if let Some(sz) = scalar_size(item) {
                    let len = usize::try_from(n).ok().and_then(|n| n.checked_mul(sz)).ok_or_else(|| Error("metadata array too long".into()))?;
                    c.take(len)?;
                } else {
                    for _ in 0..n {
                        value(c, item)?;
                    }
                }
                Value::LongArray { item_type: item, len: n }
            } else {
                Value::Array((0..n).map(|_| value(c, item)).collect::<Result<_>>()?)
            }
        }
        10 => Value::U64(c.u64()?),
        11 => Value::I64(c.u64()? as i64),
        12 => Value::F64(f64::from_le_bytes(b(c, 8)?.try_into().unwrap())),
        t => return err(format!("unknown metadata value type {t}")),
    })
}

/// Parse a GGUF header from the start of a file image.
pub fn parse_header(buf: &[u8]) -> Result<Header> {
    let mut c = Cursor { buf, pos: 0 };
    if c.take(4)? != b"GGUF" {
        return err("not a GGUF file");
    }
    let version = c.u32()?;
    if version != 2 && version != 3 {
        return err(format!("unsupported GGUF version {version}"));
    }
    let n_tensors = c.u64()?;
    let n_kv = c.u64()?;
    // counts come from the file: never reserve more than the bytes left could hold (a key/value
    // takes >= 12 bytes, a tensor entry >= 24), so a crafted count cannot abort the allocation
    let left = (buf.len() - c.pos) as u64;
    if n_kv > left / 12 || n_tensors > left / 24 {
        return err(format!("GGUF header claims {n_kv} keys / {n_tensors} tensors, more than its {left} bytes hold"));
    }
    let mut metadata = Vec::with_capacity(n_kv as usize);
    for _ in 0..n_kv {
        let key = c.string()?;
        let vt = c.u32()?;
        metadata.push((key, value(&mut c, vt)?));
    }
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = c.string()?;
        let nd = c.u32()?;
        if nd > MAX_DIMS {
            return err(format!("{name}: {nd} dimensions (ggml has at most {MAX_DIMS})"));
        }
        let shape = (0..nd).map(|_| c.u64()).collect::<Result<Vec<_>>>()?;
        // ggml keeps ne as int64
        if shape.iter().any(|&d| d > i64::MAX as u64) {
            return err(format!("{name}: dimension above 2^63"));
        }
        let ty = GgmlType(c.u32()?);
        let offset = c.u64()?;
        let (tname, blk, _) = match ty.info() {
            Some(i) => i,
            None => return err(format!("{name}: unknown ggml type id {}", ty.0)),
        };
        let t = TensorInfo { name, shape, ty, offset };
        let Some(n) = t.checked_elements().filter(|&n| n <= i64::MAX as u64) else {
            return err(format!("{}: element count overflows (shape {:?})", t.name, t.shape));
        };
        if n % blk != 0 {
            return err(format!("{}: {n} elements not a multiple of {tname} block {blk}", t.name));
        }
        if t.checked_bytes().is_none() {
            return err(format!("{}: byte size overflows", t.name));
        }
        tensors.push(t);
    }
    let mut h = Header { version, metadata, tensors, data_offset: 0, header_bytes_read: c.pos as u64 };
    let align = h.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32);
    if align == 0 || !align.is_power_of_two() {
        return err(format!("general.alignment {align} is not a power of two"));
    }
    h.data_offset = (c.pos as u64).div_ceil(align) * align;
    Ok(h)
}

/// Every tensor's data range lies inside a file of `file_bytes` bytes, with checked arithmetic
/// (REVIEW-v0.1.1 S1: plain u64 adds wrapped for an offset near 2^64 and passed the check).
/// After this, `data_offset + offset + bytes` fits in u64 and in the mapping for every tensor.
pub fn check_bounds(h: &Header, file_bytes: u64) -> Result<()> {
    for t in &h.tensors {
        let end = t.checked_bytes().and_then(|b| h.data_offset.checked_add(t.offset)?.checked_add(b));
        match end {
            Some(e) if e <= file_bytes => {}
            _ => return err(format!("{}: data beyond end of file", t.name)),
        }
    }
    Ok(())
}

/// An mmapped GGUF file plus its parsed header.
pub struct Gguf {
    pub path: PathBuf,
    pub header: Header,
    pub file_bytes: u64,
    mmap: Mmap,
    file: File,
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Gguf> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        // SAFETY: read-only private mapping of a model file we do not modify; concurrent
        // truncation by another process would be a user error (same as llama.cpp).
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error(format!("mmap {}: {e}", path.display())))?;
        let header = parse_header(&mmap)?;
        let file_bytes = mmap.len() as u64;
        check_bounds(&header, file_bytes)?;
        Ok(Gguf { path, header, file_bytes, mmap, file })
    }

    /// The raw bytes of a tensor (touches only the pages actually read by the caller).
    pub fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
        let start = (self.header.data_offset + t.offset) as usize;
        &self.mmap[start..start + t.bytes() as usize]
    }

    /// Unmap (MADV_DONTNEED) the pages of `len` bytes at `off` inside a tensor's data from this
    /// process (the page cache keeps them until [`Gguf::drop_cache`]); page-aligned inward.
    pub fn release_range(&self, t: &TensorInfo, off: usize, len: usize) {
        const PAGE: usize = 4096;
        let start = self.mmap.as_ptr() as usize + (self.header.data_offset + t.offset) as usize + off;
        let a = start.div_ceil(PAGE) * PAGE;
        let b = (start + len) / PAGE * PAGE;
        if b > a {
            // SAFETY: advisory on pages of our own read-only mapping; nothing keeps references
            // into them across this call (the caller has finished reading the range).
            unsafe { madvise(a as *mut u8, b - a, 4 /* MADV_DONTNEED */) };
        }
    }

    /// Advise the kernel that a tensor's file pages can be dropped from the page cache.
    pub fn drop_cache(&self, t: &TensorInfo) {
        use std::os::unix::io::AsRawFd;
        let start = (self.header.data_offset + t.offset) as i64;
        // SAFETY: plain advisory syscall on our own fd.
        unsafe {
            libc_fadvise(self.file.as_raw_fd(), start, t.bytes() as i64);
        }
    }
}

extern "C" {
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
}
unsafe fn libc_fadvise(fd: i32, off: i64, len: i64) {
    const POSIX_FADV_DONTNEED: i32 = 4;
    posix_fadvise(fd, off, len, POSIX_FADV_DONTNEED);
}

#[cfg(test)]
mod tests;
