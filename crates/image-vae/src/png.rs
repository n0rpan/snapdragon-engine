//! Minimal PNG support: 8-bit RGB writer (zlib "stored" blocks, filter 0) and a
//! reader for 8-bit RGB / RGBA, non-interlaced PNGs with any zlib/deflate stream
//! (stored, fixed and dynamic Huffman) and all five scanline filters.
//! Alpha is dropped on read. CRC-32 and Adler-32 are implemented here.

use std::path::Path;

use crate::Rgb;

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

const CRC_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

/// CRC-32 (ISO-HDLC, as used by PNG).
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = CRC_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// Adler-32 (zlib trailer).
pub fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

/// Writes an 8-bit RGB PNG.
pub fn write_png(path: &Path, rgb: &Rgb) -> Result<(), String> {
    let bytes = encode_png(rgb)?;
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Encodes an 8-bit RGB PNG into memory.
pub fn encode_png(rgb: &Rgb) -> Result<Vec<u8>, String> {
    let (w, h) = (rgb.width, rgb.height);
    if w == 0 || h == 0 {
        return Err("png: empty image".into());
    }
    if w > u32::MAX as usize || h > u32::MAX as usize {
        return Err("png: image too large".into());
    }
    if rgb.data.len() != w * h * 3 {
        return Err(format!("png: data length {} != {}", rgb.data.len(), w * h * 3));
    }
    let stride = w * 3;
    let mut raw = Vec::with_capacity(h * (stride + 1));
    for row in rgb.data.chunks_exact(stride) {
        raw.push(0); // filter type None
        raw.extend_from_slice(row);
    }
    let z = zlib_stored(&raw);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // depth 8, RGB, deflate, adaptive filters, no interlace

    let mut out = Vec::with_capacity(z.len() + 100);
    out.extend_from_slice(&SIGNATURE);
    write_chunk(&mut out, b"IHDR", &ihdr);
    write_chunk(&mut out, b"IDAT", &z);
    write_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// zlib stream made of stored (uncompressed) deflate blocks.
fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    if raw.is_empty() {
        out.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]);
    } else {
        let mut chunks = raw.chunks(65535).peekable();
        while let Some(c) = chunks.next() {
            let last = chunks.peek().is_none();
            out.push(if last { 1 } else { 0 });
            let len = c.len() as u16;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&(!len).to_le_bytes());
            out.extend_from_slice(c);
        }
    }
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

/// Reads an 8-bit RGB or RGBA, non-interlaced PNG.
pub fn read_png(path: &Path) -> Result<Rgb, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    decode_png(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Decodes an 8-bit RGB or RGBA, non-interlaced PNG held in memory.
pub fn decode_png(bytes: &[u8]) -> Result<Rgb, String> {
    if bytes.len() < 8 || bytes[..8] != SIGNATURE {
        return Err("png: bad signature".into());
    }
    let mut pos = 8usize;
    let mut header: Option<(usize, usize, usize)> = None; // (w, h, bpp)
    let mut idat: Vec<u8> = Vec::new();
    let mut saw_end = false;
    while pos < bytes.len() {
        if pos + 8 > bytes.len() {
            return Err("png: truncated chunk header".into());
        }
        let len = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
        let kind = &bytes[pos + 4..pos + 8];
        let data_start = pos + 8;
        let data_end = data_start
            .checked_add(len)
            .ok_or("png: chunk length overflow")?;
        if data_end + 4 > bytes.len() {
            return Err("png: truncated chunk".into());
        }
        let data = &bytes[data_start..data_end];
        let stored_crc = u32::from_be_bytes([
            bytes[data_end],
            bytes[data_end + 1],
            bytes[data_end + 2],
            bytes[data_end + 3],
        ]);
        if crc32(&bytes[pos + 4..data_end]) != stored_crc {
            return Err(format!("png: CRC mismatch in {} chunk", String::from_utf8_lossy(kind)));
        }
        match kind {
            b"IHDR" => {
                if len != 13 {
                    return Err("png: bad IHDR length".into());
                }
                let w = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
                let h = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
                let (depth, ctype, comp, filt, ilace) = (data[8], data[9], data[10], data[11], data[12]);
                if depth != 8 {
                    return Err(format!("png: bit depth {depth} not supported (8 only)"));
                }
                let bpp = match ctype {
                    2 => 3,
                    6 => 4,
                    other => return Err(format!("png: colour type {other} not supported (2 or 6)")),
                };
                if comp != 0 || filt != 0 || ilace != 0 {
                    return Err("png: unsupported compression/filter/interlace method".into());
                }
                if w == 0 || h == 0 {
                    return Err("png: zero-sized image".into());
                }
                header = Some((w, h, bpp));
            }
            b"IDAT" => {
                if header.is_none() {
                    return Err("png: IDAT before IHDR".into());
                }
                idat.extend_from_slice(data);
            }
            b"IEND" => {
                saw_end = true;
                break;
            }
            _ => {}
        }
        pos = data_end + 4;
    }
    if !saw_end {
        return Err("png: missing IEND".into());
    }
    let (w, h, bpp) = header.ok_or("png: missing IHDR")?;
    if idat.is_empty() {
        return Err("png: missing IDAT".into());
    }
    let raw = zlib_decompress(&idat)?;
    let stride = w * bpp;
    if raw.len() < h * (stride + 1) {
        return Err(format!("png: image data too short ({} < {})", raw.len(), h * (stride + 1)));
    }
    let mut data = vec![0u8; w * h * 3];
    let mut prev = vec![0u8; stride];
    let mut cur = vec![0u8; stride];
    for y in 0..h {
        let line = &raw[y * (stride + 1)..(y + 1) * (stride + 1)];
        let ftype = line[0];
        cur.copy_from_slice(&line[1..]);
        unfilter_row(ftype, &mut cur, &prev, bpp)?;
        let dst = &mut data[y * w * 3..(y + 1) * w * 3];
        for x in 0..w {
            dst[3 * x..3 * x + 3].copy_from_slice(&cur[x * bpp..x * bpp + 3]);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    Ok(Rgb {
        width: w,
        height: h,
        data,
    })
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i16 + b as i16 - c as i16;
    let pa = (p - a as i16).abs();
    let pb = (p - b as i16).abs();
    let pc = (p - c as i16).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Reverses one scanline filter in place. `cur` holds the filtered bytes on entry.
fn unfilter_row(ftype: u8, cur: &mut [u8], prev: &[u8], bpp: usize) -> Result<(), String> {
    match ftype {
        0 => {}
        1 => {
            for i in bpp..cur.len() {
                cur[i] = cur[i].wrapping_add(cur[i - bpp]);
            }
        }
        2 => {
            for i in 0..cur.len() {
                cur[i] = cur[i].wrapping_add(prev[i]);
            }
        }
        3 => {
            for i in 0..cur.len() {
                let left = if i >= bpp { cur[i - bpp] as u16 } else { 0 };
                let avg = ((left + prev[i] as u16) / 2) as u8;
                cur[i] = cur[i].wrapping_add(avg);
            }
        }
        4 => {
            for i in 0..cur.len() {
                let (a, c) = if i >= bpp { (cur[i - bpp], prev[i - bpp]) } else { (0, 0) };
                cur[i] = cur[i].wrapping_add(paeth(a, prev[i], c));
            }
        }
        other => return Err(format!("png: invalid filter type {other}")),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// zlib / DEFLATE (RFC 1950 / 1951) decoder
// ---------------------------------------------------------------------------

/// Decodes a zlib stream and verifies its Adler-32 trailer when present.
pub fn zlib_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 2 {
        return Err("zlib: stream too short".into());
    }
    let (cmf, flg) = (data[0], data[1]);
    if cmf & 0x0F != 8 || (cmf >> 4) > 7 {
        return Err("zlib: unsupported compression method".into());
    }
    if (cmf as u16 * 256 + flg as u16) % 31 != 0 {
        return Err("zlib: header check failed".into());
    }
    if flg & 0x20 != 0 {
        return Err("zlib: preset dictionary not supported".into());
    }
    let mut br = Bits {
        data,
        pos: 2,
        buf: 0,
        cnt: 0,
    };
    let mut out = Vec::with_capacity(data.len() * 4);
    inflate(&mut br, &mut out)?;
    // The bit reader never reads past the byte holding the last used bit.
    if br.pos + 4 <= data.len() {
        let stored = u32::from_be_bytes([data[br.pos], data[br.pos + 1], data[br.pos + 2], data[br.pos + 3]]);
        if stored != adler32(&out) {
            return Err("zlib: Adler-32 mismatch".into());
        }
    }
    Ok(out)
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u32,
    cnt: u32,
}

impl Bits<'_> {
    /// Reads `n` (<= 16) bits, LSB first.
    fn bits(&mut self, n: u32) -> Result<u32, String> {
        while self.cnt < n {
            let b = *self
                .data
                .get(self.pos)
                .ok_or("deflate: unexpected end of stream")?;
            self.buf |= (b as u32) << self.cnt;
            self.cnt += 8;
            self.pos += 1;
        }
        let v = self.buf & ((1u32 << n) - 1);
        self.buf >>= n;
        self.cnt -= n;
        Ok(v)
    }

    /// Discards bits up to the next byte boundary. Because `bits` never loads a byte
    /// it does not need, fewer than 8 bits are ever pending, so this drops only them.
    fn align(&mut self) {
        self.buf = 0;
        self.cnt = 0;
    }
}

struct Huffman {
    count: [u16; 16],
    symbol: Vec<u16>,
}

impl Huffman {
    /// Canonical Huffman table from code lengths (0 = unused). Over-subscribed sets
    /// are rejected; incomplete sets are accepted (valid for single-code distance trees).
    fn build(lengths: &[u8]) -> Result<Huffman, String> {
        let mut count = [0u16; 16];
        for &l in lengths {
            if l > 15 {
                return Err("deflate: code length > 15".into());
            }
            count[l as usize] += 1;
        }
        count[0] = 0;
        let mut left: i32 = 1;
        for len in 1..16 {
            left <<= 1;
            left -= count[len] as i32;
            if left < 0 {
                return Err("deflate: over-subscribed Huffman code".into());
            }
        }
        let mut offs = [0u16; 16];
        for len in 1..15 {
            offs[len + 1] = offs[len] + count[len];
        }
        let mut symbol = vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbol[offs[l as usize] as usize] = sym as u16;
                offs[l as usize] += 1;
            }
        }
        Ok(Huffman { count, symbol })
    }

    fn decode(&self, br: &mut Bits) -> Result<u16, String> {
        let mut code: i32 = 0;
        let mut first: i32 = 0;
        let mut index: i32 = 0;
        for len in 1..16 {
            code |= br.bits(1)? as i32;
            let count = self.count[len] as i32;
            if code - count < first {
                let idx = index + (code - first);
                return self
                    .symbol
                    .get(idx as usize)
                    .copied()
                    .ok_or_else(|| "deflate: bad Huffman symbol index".to_string());
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err("deflate: invalid Huffman code".into())
    }
}

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
];
const CLEN_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

fn inflate(br: &mut Bits, out: &mut Vec<u8>) -> Result<(), String> {
    loop {
        let last = br.bits(1)?;
        let btype = br.bits(2)?;
        match btype {
            0 => {
                br.align();
                let len = br.bits(16)? as usize;
                let nlen = br.bits(16)? as usize;
                if len != (!nlen & 0xFFFF) {
                    return Err("deflate: stored block LEN/NLEN mismatch".into());
                }
                let end = br.pos + len;
                if end > br.data.len() {
                    return Err("deflate: stored block past end of stream".into());
                }
                out.extend_from_slice(&br.data[br.pos..end]);
                br.pos = end;
            }
            1 => {
                let mut lens = [0u8; 288];
                for (i, l) in lens.iter_mut().enumerate() {
                    *l = match i {
                        0..=143 => 8,
                        144..=255 => 9,
                        256..=279 => 7,
                        _ => 8,
                    };
                }
                let lit = Huffman::build(&lens)?;
                let dist = Huffman::build(&[5u8; 30])?;
                inflate_codes(br, out, &lit, &dist)?;
            }
            2 => {
                let hlit = br.bits(5)? as usize + 257;
                let hdist = br.bits(5)? as usize + 1;
                let hclen = br.bits(4)? as usize + 4;
                let mut cl_lens = [0u8; 19];
                for &idx in CLEN_ORDER.iter().take(hclen) {
                    cl_lens[idx] = br.bits(3)? as u8;
                }
                let cl = Huffman::build(&cl_lens)?;
                let total = hlit + hdist;
                let mut lens = vec![0u8; total];
                let mut i = 0usize;
                while i < total {
                    let sym = cl.decode(br)?;
                    let (val, rep) = match sym {
                        0..=15 => (sym as u8, 1usize),
                        16 => {
                            if i == 0 {
                                return Err("deflate: repeat with no previous length".into());
                            }
                            (lens[i - 1], 3 + br.bits(2)? as usize)
                        }
                        17 => (0, 3 + br.bits(3)? as usize),
                        18 => (0, 11 + br.bits(7)? as usize),
                        _ => return Err("deflate: bad code-length symbol".into()),
                    };
                    if i + rep > total {
                        return Err("deflate: code lengths overflow".into());
                    }
                    for l in &mut lens[i..i + rep] {
                        *l = val;
                    }
                    i += rep;
                }
                if lens[256] == 0 {
                    return Err("deflate: no end-of-block code".into());
                }
                let lit = Huffman::build(&lens[..hlit])?;
                let dist = Huffman::build(&lens[hlit..])?;
                inflate_codes(br, out, &lit, &dist)?;
            }
            _ => return Err("deflate: invalid block type".into()),
        }
        if last == 1 {
            return Ok(());
        }
    }
}

fn inflate_codes(br: &mut Bits, out: &mut Vec<u8>, lit: &Huffman, dist: &Huffman) -> Result<(), String> {
    loop {
        let sym = lit.decode(br)? as usize;
        if sym < 256 {
            out.push(sym as u8);
        } else if sym == 256 {
            return Ok(());
        } else {
            let li = sym - 257;
            if li >= 29 {
                return Err("deflate: invalid length symbol".into());
            }
            let len = LEN_BASE[li] as usize + br.bits(LEN_EXTRA[li] as u32)? as usize;
            let di = dist.decode(br)? as usize;
            if di >= 30 {
                return Err("deflate: invalid distance symbol".into());
            }
            let d = DIST_BASE[di] as usize + br.bits(DIST_EXTRA[di] as u32)? as usize;
            if d > out.len() {
                return Err("deflate: distance too far back".into());
            }
            let start = out.len() - d;
            for k in 0..len {
                let b = out[start + k];
                out.push(b);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_and_adler_known_values() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(adler32(b""), 1);
    }

    #[test]
    fn zlib_stored_round_trip_multi_block() {
        let data: Vec<u8> = (0..150_000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let z = zlib_stored(&data);
        assert_eq!(zlib_decompress(&z).unwrap(), data);
        assert_eq!(zlib_decompress(&zlib_stored(&[])).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn paeth_matches_spec_example() {
        assert_eq!(paeth(10, 20, 10), 20);
        assert_eq!(paeth(10, 10, 20), 10);
        assert_eq!(paeth(0, 0, 0), 0);
    }

    #[test]
    fn encode_decode_round_trip() {
        let w = 37usize;
        let h = 11usize;
        let data: Vec<u8> = (0..w * h * 3).map(|i| ((i * 31) % 251) as u8).collect();
        let rgb = Rgb {
            width: w,
            height: h,
            data,
        };
        let bytes = encode_png(&rgb).unwrap();
        assert_eq!(decode_png(&bytes).unwrap(), rgb);
    }

    #[test]
    fn rejects_corrupt_crc() {
        let rgb = Rgb {
            width: 2,
            height: 2,
            data: vec![1; 12],
        };
        let mut bytes = encode_png(&rgb).unwrap();
        let n = bytes.len();
        bytes[n - 6] ^= 0xFF; // inside IEND/IDAT CRC region
        assert!(decode_png(&bytes).is_err());
    }
}
