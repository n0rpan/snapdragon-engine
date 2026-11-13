//! Persistent prefix store: engine states of prompt prefixes on disk, reused across server
//! restarts and conversations (the Rust counterpart of the controller's llama.cpp
//! PrefixStore, `snapdragon_ai_lab/runtime/prefix_store.py`).
//!
//! What is stored: only the system prompt + tool definitions ("anchor" prefix, the tokens
//! before the first user message, cut down to the prefill ubatch grid). No user turns, tool
//! results or generated text are ever written, the same scope as the controller's store.
//!
//! Identity: an entry is usable only by a server with the same identity string, made of the
//! model (`OXE_MODEL_SHA256`, as the controller sets it, else path + size + mtime), the
//! server binary (SHA-256 of the running executable), the engine's state layout (MTP head
//! or not) and the prefill ubatch. The file name is SHA-256(identity, token ids); the file
//! repeats both and they are compared in full before a load.
//!
//! Exactness: entries are saved at ubatch-grid positions of a state that was prefilled from
//! position 0 on the same grid, and a load continues on that grid, so a request that loads
//! an entry computes bit for bit what a cold server computes for the same prompt.
//!
//! Layout: `DIR/<key>.bin` (mode 0600, DIR 0700) = magic `OXEPFX02`, u32 identity length,
//! identity, u64 n_tokens, n_tokens u32 token ids, u64 state length, u64 state checksum
//! ([`Check64`]), the engine state ([`Engine::save_state`]). Entries of the older `OXEPFX01`
//! layout (no length / checksum) are still read. A load whose state length, checksum or trailing
//! bytes do not match is an error: the entry is deleted and the caller resets the engine (a
//! flipped bit inside a Flash state would otherwise load silently).
//! Writes go to `<key>.pending` and are renamed when complete; size is capped with LRU
//! eviction by mtime (a load touches the file). After the write (fsync) and after a load the
//! file's page cache is dropped (POSIX_FADV_DONTNEED): a 285 MB Flash state would otherwise sit
//! in the engine scope's memory charge next to the expert cache.

use crate::engine::Engine;
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

const MAGIC: &[u8; 8] = b"OXEPFX01";
const MAGIC2: &[u8; 8] = b"OXEPFX02";

/// Streaming 64-bit checksum (moved to `model::check` so the kernel-path probe can use it).
pub use model::check::Check64;

pub fn check64(d: &[u8]) -> u64 {
    let mut c = Check64::default();
    c.update(d);
    c.finish()
}

/// Reader over exactly `left` state bytes that checksums what the engine reads.
struct CheckedReader<'a> {
    r: &'a mut dyn Read,
    left: u64,
    sum: Check64,
}

impl Read for CheckedReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = (buf.len() as u64).min(self.left) as usize;
        if n == 0 {
            return Ok(0);
        }
        let k = self.r.read(&mut buf[..n])?;
        self.sum.update(&buf[..k]);
        self.left -= k as u64;
        Ok(k)
    }
}

/// File writer that writes behind: every `DROP_BEHIND` bytes it waits for that window's writeback
/// and drops its page cache, so a multi-GiB state never sits in the page cache (charged to the
/// model's memory cgroup: reclaim pressure / PSI next to the weights). LONG-CTX.
struct DropBehind {
    f: fs::File,
    off: u64,
    done: u64,
}

const DROP_BEHIND: u64 = 64 << 20;

impl Write for DropBehind {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.f.write(buf)?;
        self.off += n as u64;
        if self.off - self.done >= DROP_BEHIND {
            use std::os::fd::AsRawFd;
            let (fd, start, len) = (self.f.as_raw_fd(), self.done as i64, (self.off - self.done) as i64);
            // SAFETY: valid fd, a range of this file; advisory calls (errors ignored: best effort).
            unsafe {
                libc::sync_file_range(fd, start, len, libc::SYNC_FILE_RANGE_WAIT_BEFORE | libc::SYNC_FILE_RANGE_WRITE | libc::SYNC_FILE_RANGE_WAIT_AFTER);
                libc::posix_fadvise(fd, start, len, libc::POSIX_FADV_DONTNEED);
            }
            self.done = self.off;
        }
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.f.flush()
    }
}

/// Writer that checksums (and counts, `sum.len`) what the engine writes.
struct CheckedWriter<'a> {
    w: &'a mut dyn Write,
    sum: Check64,
}

impl Write for CheckedWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.w.write(buf)?;
        self.sum.update(&buf[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

/// MemAvailable from /proc/meminfo.
pub fn proc_mem_available() -> Option<u64> {
    let s = fs::read_to_string("/proc/meminfo").ok()?;
    let l = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    l.split_whitespace().nth(1)?.parse::<u64>().ok()?.checked_mul(1024)
}

/// Drop the page cache of `f` (whole file). Best effort.
fn drop_pages(f: &fs::File) {
    use std::os::fd::AsRawFd;
    // SAFETY: valid open file descriptor.
    unsafe {
        libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

/// Minimal SHA-256 (FIPS 180-4).
#[derive(Clone)]
pub struct Sha256 {
    h: [u32; 8],
    buf: Vec<u8>,
    len: u64,
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Default for Sha256 {
    fn default() -> Self {
        Sha256 { h: [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19], buf: Vec::with_capacity(64), len: 0 }
    }
}

impl Sha256 {
    fn block(&mut self, b: &[u8]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(b[4 * i..4 * i + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b_, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b_) ^ (a & c) ^ (b_ & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b_;
            b_ = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in self.h.iter_mut().zip([a, b_, c, d, e, f, g, h]) {
            *x = x.wrapping_add(y);
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len += data.len() as u64;
        if !self.buf.is_empty() {
            let n = (64 - self.buf.len()).min(data.len());
            self.buf.extend_from_slice(&data[..n]);
            data = &data[n..];
            if self.buf.len() == 64 {
                let b = std::mem::take(&mut self.buf);
                self.block(&b);
                self.buf = b;
                self.buf.clear();
            }
        }
        while data.len() >= 64 {
            self.block(&data[..64]);
            data = &data[64..];
        }
        self.buf.extend_from_slice(data);
    }

    pub fn hex(mut self) -> String {
        let bits = self.len * 8;
        let mut pad = vec![0x80u8];
        while (self.buf.len() + pad.len()) % 64 != 56 {
            pad.push(0);
        }
        pad.extend_from_slice(&bits.to_be_bytes());
        let len = self.len;
        self.update(&pad);
        self.len = len;
        self.h.iter().map(|x| format!("{x:08x}")).collect()
    }
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut s = Sha256::default();
    s.update(data);
    s.hex()
}

/// SHA-256 of a file.
pub fn sha256_file(p: &Path) -> std::io::Result<String> {
    let mut f = BufReader::with_capacity(1 << 20, fs::File::open(p)?);
    let mut s = Sha256::default();
    let mut b = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            return Ok(s.hex());
        }
        s.update(&b[..n]);
    }
}

fn tokens_bytes(t: &[u32]) -> Vec<u8> {
    t.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Entry {
    key: String,
    tokens: Vec<u32>,
}

pub struct PrefixStore {
    pub dir: PathBuf,
    pub max_bytes: u64,
    /// smallest prefix worth storing (tokens)
    pub min_tokens: usize,
    /// FLASH-PREFIX-2: skip a save unless MemAvailable >= this floor + 2 x the entry (the state buffer + its
    /// page cache until the write drops it). 0 = always save (the MoE default).
    pub save_floor_bytes: u64,
    /// MemAvailable reader (tests replace it)
    pub mem_available: fn() -> Option<u64>,
    identity: String,
    entries: Arc<Mutex<Vec<Entry>>>,
    writers: Vec<std::thread::JoinHandle<()>>,
}

/// Outcome of [`PrefixStore::load`].
pub struct Loaded {
    pub n: usize,
    pub ms: f64,
}

/// fsync a directory (after a rename into it), best effort.
fn fsync_dir(p: &Path) {
    if let Ok(f) = fs::File::open(p) {
        let _ = f.sync_all();
    }
}

impl PrefixStore {
    /// Open (create) `dir` for entries of `identity`; drops stale `.pending` files, indexes
    /// the entries of this identity and applies the size cap.
    pub fn open(dir: &Path, identity: String, max_bytes: u64, min_tokens: usize) -> Result<PrefixStore, String> {
        fs::create_dir_all(dir).map_err(|e| format!("prefix store {}: {e}", dir.display()))?;
        let _ = fs::set_permissions(dir, fs::Permissions::from_mode(0o700));
        let mut entries = Vec::new();
        for de in fs::read_dir(dir).map_err(|e| format!("prefix store {}: {e}", dir.display()))?.flatten() {
            let p = de.path();
            match p.extension().and_then(|e| e.to_str()) {
                Some("pending") => {
                    // only stale ones: a fresh .pending may be another server's write in progress
                    // in a shared directory (W4-4 c)
                    let stale = fs::metadata(&p).ok().and_then(|m| m.modified().ok()).and_then(|t| t.elapsed().ok()).map(|a| a.as_secs() > 600).unwrap_or(false);
                    if stale {
                        let _ = fs::remove_file(&p);
                    }
                }
                Some("bin") => {
                    if let Ok(Some(t)) = read_header(&p, &identity) {
                        let key = p.file_stem().unwrap().to_string_lossy().into_owned();
                        if key == entry_key(&identity, &t) {
                            entries.push(Entry { key, tokens: t });
                        }
                    }
                }
                _ => {}
            }
        }
        let st = PrefixStore {
            dir: dir.to_path_buf(),
            max_bytes,
            min_tokens,
            save_floor_bytes: 0,
            mem_available: proc_mem_available,
            identity,
            entries: Arc::new(Mutex::new(entries)),
            writers: Vec::new(),
        };
        prune(&st.dir, st.max_bytes, None, &st.entries);
        Ok(st)
    }

    /// Delete this store's entries not used for `ttl` (file mtime; a load touches it). Other
    /// identities' files in the same directory are never touched (W4-4 c). Returns how many.
    pub fn prune_older_than(&mut self, ttl: std::time::Duration) -> usize {
        let mut es = self.entries.lock().unwrap();
        let mut gone = 0;
        let older = |p: &Path| fs::metadata(p).ok().and_then(|m| m.modified().ok()).and_then(|t| t.elapsed().ok()).map(|a| a > ttl).unwrap_or(false);
        es.retain(|e| {
            let p = self.dir.join(format!("{}.bin", e.key));
            if older(&p) && fs::remove_file(&p).is_ok() {
                gone += 1;
                false
            } else {
                true
            }
        });
        // Orphans of this model from an older build (another server sha or ubatch): every rebuild
        // starts a new identity, and nothing else would ever remove them. Only files with the
        // store's magic, the SAME model component and an mtime past the TTL (a live server of that
        // identity touches its files on load); other models' files and foreign files stay.
        let own: std::collections::HashSet<String> = es.iter().map(|e| e.key.clone()).collect();
        drop(es);
        let mine = model_of(&self.identity);
        if let (Some(mine), Ok(rd)) = (mine, fs::read_dir(&self.dir)) {
            for d in rd.flatten() {
                let p = d.path();
                if p.extension().and_then(|e| e.to_str()) != Some("bin") {
                    continue;
                }
                let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                if own.contains(&stem) || !older(&p) {
                    continue;
                }
                let same_model = read_identity(&p).as_deref().and_then(model_of) == Some(mine);
                if same_model && fs::remove_file(&p).is_ok() {
                    gone += 1;
                }
            }
        }
        gone
    }

    /// Delete this store's entries whose tokens match `f` (cache clear / session clear across the
    /// legacy stores, Codex P2/P3 review 1-3). Waits for pending writes first. Returns how many.
    pub fn remove_where(&mut self, f: &dyn Fn(&[u32]) -> bool) -> usize {
        self.flush();
        let mut es = self.entries.lock().unwrap();
        let before = es.len();
        es.retain(|e| {
            if f(&e.tokens) {
                let _ = fs::remove_file(self.dir.join(format!("{}.bin", e.key)));
                false
            } else {
                true
            }
        });
        before - es.len()
    }

    /// FLASH-PREFIX-2: may an entry of about `entry_bytes` be built and written now? False when a floor is set and
    /// MemAvailable (unknown = no) is below floor + 2 x entry, or while an earlier write still holds its buffer
    /// (never two ~285 MB Flash states in memory at once). A skipped save only costs the disk hit later.
    pub fn save_allowed(&mut self, entry_bytes: u64) -> Result<(), String> {
        self.writers.retain(|h| !h.is_finished());
        if self.save_floor_bytes == 0 {
            return Ok(());
        }
        if !self.writers.is_empty() {
            return Err("an earlier entry is still being written".into());
        }
        let need = entry_bytes.saturating_mul(2).saturating_add(self.save_floor_bytes);
        match (self.mem_available)() {
            Some(a) if a >= need => Ok(()),
            Some(a) => Err(format!("MemAvailable {} MiB < floor {} MiB + 2 x entry {} MiB", a >> 20, self.save_floor_bytes >> 20, entry_bytes >> 20)),
            None => Err("MemAvailable unknown".into()),
        }
    }

    pub fn n_entries(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    /// Length of the longest stored prefix of `prompt` that is <= `limit` tokens.
    pub fn best(&self, prompt: &[u32], limit: usize) -> Option<usize> {
        let es = self.entries.lock().unwrap();
        es.iter().filter(|e| e.tokens.len() <= limit && prompt.starts_with(&e.tokens)).map(|e| e.tokens.len()).max()
    }

    pub fn contains(&self, tokens: &[u32]) -> bool {
        self.entries.lock().unwrap().iter().any(|e| e.tokens == tokens)
    }

    /// Load the entry for `prompt[..n]` into `eng`. On error the entry is discarded and the
    /// engine state is undefined (the caller resets it).
    pub fn load<E: Engine>(&self, eng: &mut E, prompt: &[u32], n: usize) -> Result<Loaded, String> {
        let t0 = std::time::Instant::now();
        let key = entry_key(&self.identity, &prompt[..n]);
        let path = self.dir.join(format!("{key}.bin"));
        let r = (|| -> Result<(), String> {
            let f = fs::File::open(&path).map_err(|e| e.to_string())?;
            let file_len = f.metadata().map_err(|e| e.to_string())?.len();
            let mut r = BufReader::with_capacity(4 << 20, f);
            let (toks, check) = read_header_v(&mut r, &self.identity, file_len).map_err(|e| e.to_string())?.ok_or("identity mismatch or bad header")?;
            if toks[..] != prompt[..n] {
                return Err("token mismatch".into());
            }
            match check {
                Some((len, sum)) => {
                    // the stored state length must be exactly what follows the header in this file
                    let hdr = header_len(&self.identity, toks.len()).ok_or("header size overflow")?;
                    if file_len.checked_sub(hdr) != Some(len) {
                        return Err(format!("state length {len} does not match the file ({file_len} bytes, header {hdr})"));
                    }
                    let mut cr = CheckedReader { r: &mut r, left: len, sum: Check64::default() };
                    let res = eng.load_state(&mut cr, n).map_err(|e| e.0);
                    let (left, got) = (cr.left, cr.sum.finish());
                    res?;
                    if left != 0 {
                        return Err(format!("the engine read {left} bytes less than the stored state"));
                    }
                    if got != sum {
                        return Err("state checksum mismatch".into());
                    }
                }
                None => eng.load_state(&mut r, n).map_err(|e| e.0)?,
            }
            let mut extra = [0u8; 1];
            if r.read(&mut extra).map_err(|e| e.to_string())? != 0 {
                return Err("trailing bytes".into());
            }
            drop_pages(r.get_ref());
            Ok(())
        })();
        match r {
            Ok(()) => {
                if let Ok(f) = fs::File::options().write(true).open(&path) {
                    let _ = f.set_modified(SystemTime::now());
                }
                Ok(Loaded { n, ms: t0.elapsed().as_secs_f64() * 1e3 })
            }
            Err(e) => {
                let _ = fs::remove_file(&path);
                self.entries.lock().unwrap().retain(|x| x.key != key);
                Err(format!("prefix store: {}: {e}", path.display()))
            }
        }
    }

    /// Store `state` (bytes of [`Engine::save_state`]) for `tokens` in the background.
    pub fn save_async(&mut self, tokens: Vec<u32>, state: Vec<u8>) {
        self.writers.retain(|h| !h.is_finished());
        let (dir, identity, entries, max_bytes) = (self.dir.clone(), self.identity.clone(), self.entries.clone(), self.max_bytes);
        let h = std::thread::spawn(move || {
            let key = entry_key(&identity, &tokens);
            let pending = dir.join(format!("{key}.pending"));
            let fin = dir.join(format!("{key}.bin"));
            let w = (|| -> std::io::Result<()> {
                let f = fs::File::options().write(true).create(true).truncate(true).mode(0o600).open(&pending)?;
                let mut w = BufWriter::with_capacity(4 << 20, f);
                w.write_all(MAGIC2)?;
                w.write_all(&(identity.len() as u32).to_le_bytes())?;
                w.write_all(identity.as_bytes())?;
                w.write_all(&(tokens.len() as u64).to_le_bytes())?;
                w.write_all(&tokens_bytes(&tokens))?;
                w.write_all(&(state.len() as u64).to_le_bytes())?;
                w.write_all(&check64(&state).to_le_bytes())?;
                w.write_all(&state)?;
                let f = w.into_inner().map_err(|e| e.into_error())?;
                f.sync_all()?;
                drop_pages(&f);
                fs::rename(&pending, &fin)?;
                fsync_dir(&dir); // audit v0.1.2 N5: the rename itself survives a power cut
                Ok(())
            })();
            match w {
                Ok(()) => {
                    let mut es = entries.lock().unwrap();
                    if !es.iter().any(|e| e.key == key) {
                        es.push(Entry { key: key.clone(), tokens });
                    }
                    drop(es);
                    prune(&dir, max_bytes, Some(&key), &entries);
                }
                Err(e) => {
                    let _ = fs::remove_file(&pending);
                    eprintln!("oxe-server: prefix store write failed: {e}");
                }
            }
        });
        self.writers.push(h);
    }

    /// LONG-CTX: store an entry for `tokens` synchronously, streaming the state from `write` (the
    /// engine's `save_state`) straight into the file: no state buffer in RAM (a 128K f16 state is
    /// 2.75 GiB). Same file bytes as [`PrefixStore::save_async`] of the same state: the header is
    /// written with a zero length / checksum, both patched after the stream. The page cache of the
    /// file is dropped after the sync. Returns the state's length.
    pub fn save_stream(&mut self, tokens: &[u32], write: &mut dyn FnMut(&mut dyn Write) -> Result<(), String>) -> Result<u64, String> {
        use std::io::{Seek, SeekFrom};
        let key = entry_key(&self.identity, tokens);
        let pending = self.dir.join(format!("{key}.pending"));
        let fin = self.dir.join(format!("{key}.bin"));
        let hdr = header_len(&self.identity, tokens.len()).ok_or("header size overflow")?;
        let r = (|| -> Result<u64, String> {
            let io = |e: std::io::Error| e.to_string();
            let f = fs::File::options().write(true).create(true).truncate(true).mode(0o600).open(&pending).map_err(io)?;
            let mut w = BufWriter::with_capacity(4 << 20, DropBehind { f, off: 0, done: 0 });
            w.write_all(MAGIC2).map_err(io)?;
            w.write_all(&(self.identity.len() as u32).to_le_bytes()).map_err(io)?;
            w.write_all(self.identity.as_bytes()).map_err(io)?;
            w.write_all(&(tokens.len() as u64).to_le_bytes()).map_err(io)?;
            w.write_all(&tokens_bytes(tokens)).map_err(io)?;
            w.write_all(&[0u8; 16]).map_err(io)?;
            let mut cw = CheckedWriter { w: &mut w, sum: Check64::default() };
            write(&mut cw)?;
            let (len, sum) = (cw.sum.len, cw.sum.finish());
            let mut f = w.into_inner().map_err(|e| e.into_error().to_string())?.f;
            f.seek(SeekFrom::Start(hdr - 16)).map_err(io)?;
            f.write_all(&len.to_le_bytes()).map_err(io)?;
            f.write_all(&sum.to_le_bytes()).map_err(io)?;
            f.sync_all().map_err(io)?;
            drop_pages(&f);
            fs::rename(&pending, &fin).map_err(io)?;
            fsync_dir(&self.dir); // audit v0.1.2 N5
            Ok(len)
        })();
        match r {
            Ok(len) => {
                let mut es = self.entries.lock().unwrap();
                if !es.iter().any(|e| e.key == key) {
                    es.push(Entry { key: key.clone(), tokens: tokens.to_vec() });
                }
                drop(es);
                prune(&self.dir, self.max_bytes, Some(&key), &self.entries);
                Ok(len)
            }
            Err(e) => {
                let _ = fs::remove_file(&pending);
                Err(format!("prefix store stream write: {e}"))
            }
        }
    }

    /// Wait for background writes (tests, shutdown).
    pub fn flush(&mut self) {
        for h in self.writers.drain(..) {
            let _ = h.join();
        }
    }
}

fn entry_key(identity: &str, tokens: &[u32]) -> String {
    let mut s = Sha256::default();
    s.update(&(identity.len() as u64).to_le_bytes());
    s.update(identity.as_bytes());
    s.update(&tokens_bytes(tokens));
    s.hex()
}

fn read_header(p: &Path, identity: &str) -> std::io::Result<Option<Vec<u32>>> {
    let f = fs::File::open(p)?;
    let len = f.metadata()?.len();
    let mut r = BufReader::new(f);
    Ok(read_header_v(&mut r, identity, len)?.map(|h| h.0))
}

/// Bytes of an `OXEPFX02` header (magic .. checksum) for `identity` and `n_tokens` tokens.
fn header_len(identity: &str, n_tokens: usize) -> Option<u64> {
    (n_tokens as u64).checked_mul(4)?.checked_add(8 + 4 + identity.len() as u64 + 8 + 16)
}

/// Header of an entry: Some(tokens) if it belongs to `identity`.
/// The identity string of an entry file (magic checked), whatever identity it has.
fn read_identity(p: &Path) -> Option<String> {
    let mut r = BufReader::new(fs::File::open(p).ok()?);
    let mut m = [0u8; 8];
    r.read_exact(&mut m).ok()?;
    if &m != MAGIC && &m != MAGIC2 {
        return None;
    }
    let mut u = [0u8; 4];
    r.read_exact(&mut u).ok()?;
    let n = u32::from_le_bytes(u) as usize;
    if n == 0 || n > 4096 {
        return None;
    }
    let mut id = vec![0u8; n];
    r.read_exact(&mut id).ok()?;
    String::from_utf8(id).ok()
}

/// The `model=...` component of an identity string (see [`identity`]).
fn model_of(identity: &str) -> Option<&str> {
    identity.split(' ').find(|c| c.starts_with("model="))
}

/// Header of an entry: Some(tokens) if it belongs to `identity`, plus the state's (length, checksum) of an `OXEPFX02` entry (None:
/// an `OXEPFX01` entry, which has neither).
/// `file_len` bounds every count before it sizes an allocation (a corrupt or crafted header is a miss).
fn read_header_v(r: &mut dyn Read, identity: &str, file_len: u64) -> std::io::Result<Option<(Vec<u32>, Option<(u64, u64)>)>> {
    let mut m = [0u8; 8];
    r.read_exact(&mut m)?;
    let v2 = &m == MAGIC2;
    if &m != MAGIC && !v2 {
        return Ok(None);
    }
    let mut u = [0u8; 4];
    r.read_exact(&mut u)?;
    let n = u32::from_le_bytes(u) as usize;
    if n != identity.len() {
        return Ok(None);
    }
    let mut id = vec![0u8; n];
    r.read_exact(&mut id)?;
    if id != identity.as_bytes() {
        return Ok(None);
    }
    let mut u8b = [0u8; 8];
    r.read_exact(&mut u8b)?;
    let nt = u64::from_le_bytes(u8b);
    if nt > 1 << 22 || nt.checked_mul(4).map_or(true, |b| b > file_len) {
        return Ok(None);
    }
    let nt = nt as usize;
    let mut b = vec![0u8; nt * 4];
    r.read_exact(&mut b)?;
    let toks = b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    if !v2 {
        return Ok(Some((toks, None)));
    }
    let mut l = [0u8; 8];
    r.read_exact(&mut l)?;
    let mut c = [0u8; 8];
    r.read_exact(&mut c)?;
    Ok(Some((toks, Some((u64::from_le_bytes(l), u64::from_le_bytes(c))))))
}

/// Delete this store's least recently used entries until they fit `max_bytes`. Only files
/// indexed under this store's identity (header magic + identity checked at open, or written by
/// it) are counted or deleted: a directory shared with another model or engine is left alone
/// (W4-4 c), so the cap is per identity.
fn prune(dir: &Path, max_bytes: u64, keep: Option<&str>, entries: &Mutex<Vec<Entry>>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    let own: std::collections::HashSet<String> = entries.lock().unwrap().iter().map(|e| e.key.clone()).collect();
    let mut files: Vec<(SystemTime, u64, PathBuf)> = rd
        .flatten()
        .filter(|d| d.path().extension().and_then(|e| e.to_str()) == Some("bin"))
        .filter(|d| d.path().file_stem().map(|s| own.contains(s.to_string_lossy().as_ref())).unwrap_or(false))
        .filter_map(|d| {
            let m = d.metadata().ok()?;
            Some((m.modified().ok()?, m.len(), d.path()))
        })
        .collect();
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    files.sort();
    for (_, len, p) in files {
        if total <= max_bytes {
            break;
        }
        let stem = p.file_stem().unwrap().to_string_lossy().into_owned();
        if Some(stem.as_str()) == keep {
            continue;
        }
        if fs::remove_file(&p).is_ok() {
            total -= len;
            entries.lock().unwrap().retain(|e| e.key != stem);
        }
    }
}

/// The directory of `file` when `dir` lies inside it (both canonicalized; `dir` may not exist
/// yet: its nearest existing ancestor is used). Flash refuses a prefix store there (O-21: the
/// expert sidecar may sit on an unencrypted partition; prompts must not).
pub fn inside_dir_of(dir: &Path, file: Option<&Path>) -> Option<PathBuf> {
    let side = fs::canonicalize(file?.parent()?).ok()?;
    let mut probe = if dir.is_absolute() { dir.to_path_buf() } else { std::env::current_dir().ok()?.join(dir) };
    let mut rest = Vec::new();
    let base = loop {
        if let Ok(c) = fs::canonicalize(&probe) {
            break c;
        }
        rest.push(probe.file_name()?.to_os_string());
        probe = probe.parent()?.to_path_buf();
    };
    let full = rest.iter().rev().fold(base, |p, c| p.join(c));
    full.starts_with(&side).then_some(side)
}

/// Identity string for a model file and engine (see the module docs).
pub fn identity(model: &Path, state_id: &str, ubatch: usize) -> String {
    let model_id = model_id(model);
    let exe = std::env::current_exe().ok().and_then(|p| sha256_file(&p).ok()).unwrap_or_else(|| "unknown".into());
    format!("model={model_id} server=sha256:{exe} state={state_id} ubatch={ubatch}")
}

/// The model component of an identity: `sha256:<OXE_MODEL_SHA256>` (the controller sets it), else
/// the file's canonical path, size and mtime.
pub fn model_id(model: &Path) -> String {
    match std::env::var("OXE_MODEL_SHA256") {
        Ok(s) if s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()) => format!("sha256:{}", s.to_ascii_lowercase()),
        _ => {
            let m = fs::metadata(model).ok();
            let mtime = m.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok()).map(|d| d.as_nanos()).unwrap_or(0);
            format!("file:{}:{}:{}", fs::canonicalize(model).unwrap_or(model.to_path_buf()).display(), m.map(|m| m.len()).unwrap_or(0), mtime)
        }
    }
}
