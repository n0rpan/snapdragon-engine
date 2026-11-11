//! Minimal GGUF metadata reader that keeps full arrays (the `gguf` crate deliberately
//! truncates long arrays, and the vocab is 248K strings). Reads only the KV section,
//! sequentially, so it never touches tensor data.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub enum Meta {
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Meta>),
}

impl Meta {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Meta::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Meta::Int(v) => Some(v),
            Meta::UInt(v) => i64::try_from(v).ok(),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Meta::Bool(b) => Some(b),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Meta]> {
        match self {
            Meta::Array(a) => Some(a),
            _ => None,
        }
    }
}

fn err(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

struct R<T: Read>(T);

impl<T: Read> R<T> {
    fn bytes<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let mut b = [0u8; N];
        self.0.read_exact(&mut b)?;
        Ok(b)
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> io::Result<String> {
        let n = self.u64()?;
        if n > (1 << 30) {
            return Err(err("string too long"));
        }
        let mut v = vec![0u8; n as usize];
        self.0.read_exact(&mut v)?;
        // GGUF strings are UTF-8; keep lossy rather than failing on a bad one.
        Ok(String::from_utf8(v).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned()))
    }
    fn value(&mut self, ty: u32) -> io::Result<Meta> {
        Ok(match ty {
            0 => Meta::UInt(self.bytes::<1>()?[0] as u64),
            1 => Meta::Int(self.bytes::<1>()?[0] as i8 as i64),
            2 => Meta::UInt(u16::from_le_bytes(self.bytes()?) as u64),
            3 => Meta::Int(i16::from_le_bytes(self.bytes()?) as i64),
            4 => Meta::UInt(self.u32()? as u64),
            5 => Meta::Int(self.u32()? as i32 as i64),
            6 => Meta::Float(f32::from_le_bytes(self.bytes()?) as f64),
            7 => Meta::Bool(self.bytes::<1>()?[0] != 0),
            8 => Meta::Str(self.string()?),
            9 => {
                let it = self.u32()?;
                let n = self.u64()?;
                if n > (1 << 28) {
                    return Err(err("array too long"));
                }
                let mut v = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    v.push(self.value(it)?);
                }
                Meta::Array(v)
            }
            10 => Meta::UInt(self.u64()?),
            11 => Meta::Int(self.u64()? as i64),
            12 => Meta::Float(f64::from_le_bytes(self.bytes()?)),
            t => return Err(err(format!("unknown gguf value type {t}"))),
        })
    }
}

/// Read all metadata key/values of a GGUF file (v2/v3).
pub fn read_metadata(path: impl AsRef<Path>) -> io::Result<HashMap<String, Meta>> {
    let f = File::open(path)?;
    let mut r = R(BufReader::with_capacity(1 << 20, f));
    if &r.bytes::<4>()? != b"GGUF" {
        return Err(err("not a GGUF file"));
    }
    let version = r.u32()?;
    if version < 2 {
        return Err(err("GGUF v1 not supported"));
    }
    let _n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    let mut out = HashMap::new();
    for _ in 0..n_kv {
        let k = r.string()?;
        let ty = r.u32()?;
        let v = r.value(ty)?;
        out.insert(k, v);
    }
    Ok(out)
}
