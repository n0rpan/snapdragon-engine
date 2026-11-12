//! FLASH-PREFIX-1: the persistent prefix store with a Flash-shaped engine, model-free.
//!
//! [`HybridStub`] has the shape of the Flash state the store persists: a recurrent state (GDN-like,
//! updated by every token, cannot be rewound) + one KV row per position (attention cells = positions)
//! + the logits after the last token + an expert hot list, written in the same layout as
//! `FlashEngine::save_state` (u64 compact length, i32 token history, compact bytes, logits, hot list).
//! Its numerics are a deterministic function of the token sequence, so a load that restores anything
//! but the exact state shows up as different logit bits.
//!
//! Checked: save -> load -> identical logits (bitwise) vs a cold engine; key mismatch -> miss (model,
//! state layout / vocabulary, schedule, prefix); corrupt entries (bit flip, truncation, trailing bytes,
//! garbage header) -> error + entry deleted + engine reset, then the cold result; OXEPFX01 entries
//! still load; 0700 directory / 0600 files, no `.pending` left; LRU cap; TTL orphans; the O-21 guard.

use crate::engine::{Engine, EngineError, Result};
use crate::prefix_store::{check64, inside_dir_of, Check64, PrefixStore};
use crate::tests::scratch_dir;
use std::os::unix::fs::PermissionsExt;

const REC: usize = 512;
const KV: usize = 4;

pub(crate) struct HybridStub {
    n_ctx: usize,
    n_vocab: usize,
    toks: Vec<u32>,
    rec: Vec<f32>,
    kv: Vec<[f32; KV]>,
    logits: Vec<f32>,
    hot: Vec<(u16, u16)>,
    pub loads: usize,
}

pub(crate) struct HybridSnap {
    pos: usize,
    rec: Vec<f32>,
}

impl HybridStub {
    pub(crate) fn new(n_ctx: usize, n_vocab: usize) -> HybridStub {
        HybridStub { n_ctx, n_vocab, toks: Vec::new(), rec: vec![0.0; REC], kv: Vec::new(), logits: vec![0.0; n_vocab], hot: Vec::new(), loads: 0 }
    }
    fn step(&mut self, t: u32) {
        let pos = self.toks.len() as f32;
        let x = (t as f32 * 0.618_034 + pos * 0.001).sin();
        // "attention": the KV row of this position reads the recurrent state
        let row = [x, x * self.rec[t as usize % REC], (x + self.rec[(t as usize * 7) % REC]).cos(), pos.sqrt() * 1e-3];
        self.kv.push(row);
        // "gated delta net": every element of the recurrent state moves with every token
        for (i, r) in self.rec.iter_mut().enumerate() {
            *r = *r * 0.97 + x * ((i as f32 + 1.0) * 0.013 + x).sin() * 0.05;
        }
        self.toks.push(t);
        self.hot.push(((t % 48) as u16, (t % 512) as u16));
        self.hot.sort_unstable();
        self.hot.dedup();
    }
    fn update_logits(&mut self) {
        let mut acc = [0f32; KV];
        for r in &self.kv {
            for k in 0..KV {
                acc[k] += r[k];
            }
        }
        for (j, l) in self.logits.iter_mut().enumerate() {
            *l = self.rec[j % REC] + acc[j % KV] * 1e-3 + (j as f32) * 1e-7;
        }
    }
    pub(crate) fn logits_bits(&self) -> Vec<u32> {
        self.logits.iter().map(|x| x.to_bits()).collect()
    }
    fn compact(&self) -> Vec<u8> {
        let mut b = Vec::new();
        for r in &self.kv {
            for x in r {
                b.extend_from_slice(&x.to_le_bytes());
            }
        }
        for x in &self.rec {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }
}

impl Engine for HybridStub {
    type Snapshot = HybridSnap;
    fn n_ctx(&self) -> usize {
        self.n_ctx
    }
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }
    fn n_past(&self) -> usize {
        self.toks.len()
    }
    fn reset(&mut self) {
        self.toks.clear();
        self.kv.clear();
        self.rec.iter_mut().for_each(|x| *x = 0.0);
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        if self.toks.len() + tokens.len() > self.n_ctx {
            return Err(EngineError("context full".into()));
        }
        for &t in tokens {
            if t as usize >= self.n_vocab {
                return Err(EngineError(format!("token {t} out of vocab")));
            }
            self.step(t);
        }
        self.update_logits();
        Ok(&self.logits)
    }
    fn snapshot(&mut self) -> Result<HybridSnap> {
        Ok(HybridSnap { pos: self.toks.len(), rec: self.rec.clone() })
    }
    fn restore(&mut self, s: &HybridSnap) -> Result<()> {
        if s.pos > self.toks.len() {
            return Err(EngineError("restore ahead".into()));
        }
        self.toks.truncate(s.pos);
        self.kv.truncate(s.pos);
        self.rec.copy_from_slice(&s.rec);
        self.update_logits();
        Ok(())
    }
    fn snapshot_bytes(s: &HybridSnap) -> usize {
        4 * s.rec.len()
    }
    fn state_id(&self) -> Option<String> {
        Some(format!("hybrid-stub-v1 vocab={} rec={REC}", self.n_vocab))
    }
    fn state_size_hint(&self) -> usize {
        8 + 4 * self.toks.len() + 4 * (KV * self.kv.len() + REC) + 4 * self.n_vocab
    }
    fn save_state(&mut self, w: &mut dyn std::io::Write) -> Result<()> {
        let e = |e: std::io::Error| EngineError(format!("save state: {e}"));
        let b = self.compact();
        w.write_all(&(b.len() as u64).to_le_bytes()).map_err(e)?;
        for &t in &self.toks {
            w.write_all(&(t as i32).to_le_bytes()).map_err(e)?;
        }
        w.write_all(&b).map_err(e)?;
        for x in &self.logits {
            w.write_all(&x.to_le_bytes()).map_err(e)?;
        }
        w.write_all(&(self.hot.len() as u64).to_le_bytes()).map_err(e)?;
        for &(l, x) in &self.hot {
            w.write_all(&l.to_le_bytes()).map_err(e)?;
            w.write_all(&x.to_le_bytes()).map_err(e)?;
        }
        Ok(())
    }
    fn load_state(&mut self, r: &mut dyn std::io::Read, n_pos: usize) -> Result<()> {
        let e = |e: std::io::Error| EngineError(format!("load state: {e}"));
        let mut u = [0u8; 8];
        r.read_exact(&mut u).map_err(e)?;
        let n = u64::from_le_bytes(u) as usize;
        // as FlashEngine/load_compact: the compact size must match the layout at n_pos
        if n != 4 * (KV * n_pos + REC) || n_pos > self.n_ctx {
            return Err(EngineError("compact state size does not match this context".into()));
        }
        let mut tb = vec![0u8; n_pos * 4];
        r.read_exact(&mut tb).map_err(e)?;
        let mut b = vec![0u8; n];
        r.read_exact(&mut b).map_err(e)?;
        let mut lb = vec![0u8; self.n_vocab * 4];
        r.read_exact(&mut lb).map_err(e)?;
        r.read_exact(&mut u).map_err(e)?;
        let nh = u64::from_le_bytes(u) as usize;
        if nh > 48 * 512 {
            return Err(EngineError("hot set: bad size".into()));
        }
        let mut hb = vec![0u8; nh * 4];
        r.read_exact(&mut hb).map_err(e)?;
        let f = |c: &[u8]| f32::from_le_bytes(c.try_into().unwrap());
        self.toks = tb.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect();
        let (kvb, recb) = b.split_at(4 * KV * n_pos);
        self.kv = kvb.chunks(4 * KV).map(|c| [f(&c[0..4]), f(&c[4..8]), f(&c[8..12]), f(&c[12..16])]).collect();
        self.rec = recb.chunks(4).map(f).collect();
        self.logits = lb.chunks(4).map(f).collect();
        self.hot = hb.chunks(4).map(|c| (u16::from_le_bytes([c[0], c[1]]), u16::from_le_bytes([c[2], c[3]]))).collect();
        self.loads += 1;
        Ok(())
    }
}

const ID: &str = "model=sha256:aaaa server=sha256:bbbb state=hybrid-stub-v1 vocab=4096 rec=512 sched=messages ubatch=4096";

fn prompt(n: usize, salt: u32) -> Vec<u32> {
    (0..n as u32).map(|i| i.wrapping_mul(2654435761u32.wrapping_add(salt)) % 4000 + 10).collect()
}

/// Cold engine A computes the anchor and writes it; a new engine B (a restarted process) loads it.
fn store_anchor(dir: &std::path::Path, id: &str, p: &[u32], anchor: usize) -> HybridStub {
    let mut a = HybridStub::new(16384, 4096);
    a.prefill(&p[..anchor]).unwrap();
    let mut st = PrefixStore::open(dir, id.into(), 1 << 30, 64).unwrap();
    let buf = a.save_state_vec().unwrap();
    st.save_async(p[..anchor].to_vec(), buf);
    st.flush();
    a
}

fn entries(dir: &std::path::Path, ext: &str) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(dir).unwrap().flatten().map(|d| d.path()).filter(|p| p.extension().map_or(false, |e| e == ext)).collect()
}

/// save -> load -> identical logits: B (loaded anchor + the rest of the prompt) == C (cold, whole
/// prompt), bit for bit, and B right after the load == A at the anchor (logits included in the state).
#[test]
fn flash_prefix_save_load_identical_logits() {
    let dir = scratch_dir("fp-identity");
    let p = prompt(700, 1);
    let anchor = 650;
    let a = store_anchor(&dir, ID, &p, anchor);
    let st = PrefixStore::open(&dir, ID.into(), 1 << 30, 64).unwrap();
    assert_eq!(st.n_entries(), 1);
    assert_eq!(st.best(&p, p.len() - 1), Some(anchor));
    let mut b = HybridStub::new(16384, 4096);
    let l = st.load(&mut b, &p, anchor).unwrap();
    assert_eq!((l.n, b.n_past(), b.loads), (anchor, anchor, 1));
    assert_eq!(b.logits_bits(), a.logits_bits(), "the loaded logits are the saved ones");
    assert_eq!(b.hot, a.hot, "the hot list round-trips");
    b.prefill(&p[anchor..]).unwrap();
    let mut c = HybridStub::new(16384, 4096);
    c.prefill(&p).unwrap();
    assert_eq!(b.logits_bits(), c.logits_bits(), "loaded + rest == cold, bitwise");
    assert_eq!(b.rec.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), c.rec.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    // greedy continuation: same tokens for 16 steps
    let (mut tb, mut tc) = (Vec::new(), Vec::new());
    for _ in 0..16 {
        let nb = crate::sampling::argmax(&b.logits);
        let nc = crate::sampling::argmax(&c.logits);
        tb.push(nb);
        tc.push(nc);
        b.decode(nb).unwrap();
        c.decode(nc).unwrap();
    }
    assert_eq!(tb, tc);
    // save_state_vec (the store's path) == save_state (the session cache's path)
    let mut x = HybridStub::new(16384, 4096);
    x.prefill(&p[..100]).unwrap();
    let mut via_write = Vec::new();
    x.save_state(&mut via_write).unwrap();
    assert_eq!(x.save_state_vec().unwrap(), via_write);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Key mismatch -> miss: another model sha, another state layout / vocabulary (tokenizer), another
/// prefill schedule, another server build, or a prompt that differs in one token inside the anchor.
#[test]
fn flash_prefix_key_mismatch_is_a_miss() {
    let dir = scratch_dir("fp-keys");
    let p = prompt(700, 2);
    store_anchor(&dir, ID, &p, 650);
    for other in [
        ID.replace("sha256:aaaa", "sha256:aaab"),
        ID.replace("vocab=4096", "vocab=4097"),
        ID.replace(" sched=messages", ""),
        ID.replace("ubatch=4096", "ubatch=256"),
        ID.replace("server=sha256:bbbb", "server=sha256:cccc"),
    ] {
        let st = PrefixStore::open(&dir, other.clone(), 1 << 30, 64).unwrap();
        assert_eq!(st.n_entries(), 0, "{other}");
        assert_eq!(st.best(&p, p.len() - 1), None, "{other}");
    }
    let st = PrefixStore::open(&dir, ID.into(), 1 << 30, 64).unwrap();
    let mut q = p.clone();
    q[300] ^= 1;
    assert_eq!(st.best(&q, q.len() - 1), None, "one token differs inside the anchor");
    assert_eq!(st.best(&p[..650], 649), None, "a prompt that ends at the anchor (limit = len - 1)");
    assert_eq!(st.best(&p, p.len() - 1), Some(650));
    // a file renamed to another key is not indexed (name = hash of identity + tokens)
    let f = entries(&dir, "bin").pop().unwrap();
    std::fs::rename(&f, dir.join(format!("{}.bin", "0".repeat(64)))).unwrap();
    assert_eq!(PrefixStore::open(&dir, ID.into(), 1 << 30, 64).unwrap().n_entries(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Corrupt entries: a flipped bit inside the state (which the engine itself cannot see), a truncated
/// file, trailing bytes and a garbage header. Each load fails, deletes the entry and leaves the
/// caller to reset the engine; the request then computes the cold result.
#[test]
fn flash_prefix_corrupt_entry_resets() {
    let p = prompt(700, 3);
    let mut cold = HybridStub::new(16384, 4096);
    cold.prefill(&p).unwrap();
    type Damage = fn(&std::path::Path);
    let flip: Damage = |f| {
        let mut b = std::fs::read(f).unwrap();
        let i = b.len() / 2;
        b[i] ^= 0x10;
        std::fs::write(f, b).unwrap();
    };
    let cut: Damage = |f| {
        let n = std::fs::metadata(f).unwrap().len();
        std::fs::OpenOptions::new().write(true).open(f).unwrap().set_len(n - 3).unwrap();
    };
    let extra: Damage = |f| {
        use std::io::Write;
        std::fs::OpenOptions::new().append(true).open(f).unwrap().write_all(&[0u8; 4]).unwrap();
    };
    for (name, damage) in [("flip", flip), ("cut", cut), ("extra", extra)] {
        let dir = scratch_dir(&format!("fp-corrupt-{name}"));
        store_anchor(&dir, ID, &p, 650);
        let f = entries(&dir, "bin").pop().unwrap();
        damage(&f);
        let st = PrefixStore::open(&dir, ID.into(), 1 << 30, 64).unwrap();
        assert_eq!(st.n_entries(), 1, "{name}: the header is intact");
        let mut b = HybridStub::new(16384, 4096);
        let err = st.load(&mut b, &p, 650).err().unwrap_or_else(|| panic!("{name}: loaded a damaged entry"));
        assert!(!f.exists(), "{name}: entry deleted ({err})");
        assert_eq!(st.n_entries(), 0, "{name}");
        // generate.rs::load_stored resets the engine after a failed load
        b.reset();
        b.prefill(&p).unwrap();
        assert_eq!(b.logits_bits(), cold.logits_bits(), "{name}: cold result after the reset");
        let _ = std::fs::remove_dir_all(&dir);
    }
    // garbage header: never indexed, never deleted by this store (not an entry)
    let dir = scratch_dir("fp-corrupt-hdr");
    std::fs::create_dir_all(&dir).unwrap();
    let junk = dir.join(format!("{}.bin", "1".repeat(64)));
    std::fs::write(&junk, b"OXEPFX02\xff\xff\xff\xffgarbage").unwrap();
    let st = PrefixStore::open(&dir, ID.into(), 1, 64).unwrap();
    assert_eq!(st.n_entries(), 0);
    assert!(junk.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Entries written before FLASH-PREFIX-1 (`OXEPFX01`: no state length / checksum) still load.
#[test]
fn flash_prefix_reads_v1_entries() {
    let dir = scratch_dir("fp-v1");
    std::fs::create_dir_all(&dir).unwrap();
    let p = prompt(300, 4);
    let mut a = HybridStub::new(16384, 4096);
    a.prefill(&p[..256]).unwrap();
    let state = a.save_state_vec().unwrap();
    let toks: Vec<u8> = p[..256].iter().flat_map(|t| t.to_le_bytes()).collect();
    let mut f = Vec::new();
    f.extend_from_slice(b"OXEPFX01");
    f.extend_from_slice(&(ID.len() as u32).to_le_bytes());
    f.extend_from_slice(ID.as_bytes());
    f.extend_from_slice(&256u64.to_le_bytes());
    f.extend_from_slice(&toks);
    f.extend_from_slice(&state);
    // the key the store computes: write once with v2 to learn the name, then replace the content
    let mut st = PrefixStore::open(&dir, ID.into(), 1 << 30, 64).unwrap();
    st.save_async(p[..256].to_vec(), state.clone());
    st.flush();
    let name = entries(&dir, "bin").pop().unwrap();
    std::fs::write(&name, &f).unwrap();
    let st = PrefixStore::open(&dir, ID.into(), 1 << 30, 64).unwrap();
    assert_eq!(st.n_entries(), 1);
    let mut b = HybridStub::new(16384, 4096);
    st.load(&mut b, &p, 256).unwrap();
    assert_eq!(b.logits_bits(), a.logits_bits());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Permissions: the directory becomes 0700 even when it existed as 0755, entries are 0600, and no
/// `.pending` file stays behind.
#[test]
fn flash_prefix_permissions() {
    let dir = scratch_dir("fp-perm");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    store_anchor(&dir, ID, &prompt(300, 5), 256);
    assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
    let bins = entries(&dir, "bin");
    assert_eq!(bins.len(), 1);
    for f in &bins {
        assert_eq!(std::fs::metadata(f).unwrap().permissions().mode() & 0o777, 0o600);
    }
    assert!(entries(&dir, "pending").is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Bounded budget: LRU by mtime (a load touches the entry), the entry just written is kept, and the
/// TTL prune removes this identity's stale entries plus the same model's older-build orphans only.
#[test]
fn flash_prefix_budget_lru_and_ttl() {
    let dir = scratch_dir("fp-lru");
    let ps: Vec<Vec<u32>> = (0..3).map(|k| prompt(300, 10 + k)).collect();
    let one = {
        store_anchor(&dir, ID, &ps[0], 256);
        std::fs::metadata(entries(&dir, "bin")[0].clone()).unwrap().len()
    };
    let cap = 2 * one + one / 2; // room for two entries
    let mut st = PrefixStore::open(&dir, ID.into(), cap, 64).unwrap();
    let mut a = HybridStub::new(16384, 4096);
    a.prefill(&ps[1][..256]).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    st.save_async(ps[1][..256].to_vec(), a.save_state_vec().unwrap());
    st.flush();
    assert_eq!(st.n_entries(), 2);
    // use entry 0 (touches it), then write a third: entry 1 is now the least recently used
    std::thread::sleep(std::time::Duration::from_millis(20));
    st.load(&mut HybridStub::new(16384, 4096), &ps[0], 256).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let mut c = HybridStub::new(16384, 4096);
    c.prefill(&ps[2][..256]).unwrap();
    st.save_async(ps[2][..256].to_vec(), c.save_state_vec().unwrap());
    st.flush();
    assert_eq!(st.n_entries(), 2);
    assert_eq!(st.best(&ps[0], 299), Some(256), "recently loaded entry kept");
    assert_eq!(st.best(&ps[2], 299), Some(256), "new entry kept");
    assert_eq!(st.best(&ps[1], 299), None, "LRU entry evicted");
    let total: u64 = entries(&dir, "bin").iter().map(|f| std::fs::metadata(f).unwrap().len()).sum();
    assert!(total <= cap);
    // TTL: an older build's orphan of the same model goes, another model's entry stays
    let old_build = ID.replace("server=sha256:bbbb", "server=sha256:0000");
    let other_model = ID.replace("sha256:aaaa", "sha256:ffff");
    store_anchor(&dir, &old_build, &ps[0], 256);
    store_anchor(&dir, &other_model, &ps[0], 256);
    std::thread::sleep(std::time::Duration::from_millis(20));
    let gone = st.prune_older_than(std::time::Duration::from_millis(1));
    assert_eq!(gone, 3, "two own entries + the old build's orphan");
    assert_eq!(entries(&dir, "bin").len(), 1, "the other model's entry survives");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The checksum: streaming == one shot at any split, sensitive to one bit and to the length.
#[test]
fn flash_prefix_check64() {
    let d: Vec<u8> = (0..10_000u32).map(|i| (i * 31 % 251) as u8).collect();
    let whole = check64(&d);
    for split in [0, 1, 7, 8, 9, 63, 4096, 9999, 10_000] {
        let mut c = Check64::default();
        c.update(&d[..split]);
        for ch in d[split..].chunks(13) {
            c.update(ch);
        }
        assert_eq!(c.finish(), whole, "split {split}");
    }
    let mut e = d.clone();
    e[5000] ^= 1;
    assert_ne!(check64(&e), whole);
    let mut z = d.clone();
    z.push(0);
    assert_ne!(check64(&z), whole);
    assert_ne!(check64(&[]), check64(&[0]));
}

/// O-21 guard (oxe-server --engine flash): a prefix directory inside the expert sidecar's directory is
/// refused, also when it does not exist yet; one elsewhere is accepted.
#[test]
fn flash_prefix_refuses_the_sidecar_directory() {
    let root = scratch_dir("fp-o21");
    let side_dir = root.join("weights/flash");
    std::fs::create_dir_all(&side_dir).unwrap();
    let side = side_dir.join("experts-r4.bin");
    std::fs::write(&side, b"x").unwrap();
    assert!(inside_dir_of(&side_dir.join("prefix"), Some(&side)).is_some());
    assert!(inside_dir_of(&side_dir.join("a/b/c"), Some(&side)).is_some(), "not yet created");
    assert!(inside_dir_of(&side_dir, Some(&side)).is_some());
    assert!(inside_dir_of(&root.join("home/.cache/prefix-flash"), Some(&side)).is_none());
    assert!(inside_dir_of(&root.join("weights"), Some(&side)).is_none(), "the parent of the sidecar dir");
    assert!(inside_dir_of(&root.join("x"), None).is_none(), "no sidecar given");
    let _ = std::fs::remove_dir_all(&root);
}

/// Through `Worker::generate` (message-boundary schedule, Flash's 4096 ubatch): a cold process stores
/// the anchor at the first user message; a restarted process loads it for a new question and returns
/// the same tokens and final logits as a cold process (needs only the tokenizer of a local GGUF).
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn flash_prefix_worker_restart_identical() {
    let Some(m) = crate::tests::model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let dir = scratch_dir("fp-worker");
    let nv = tok.n_vocab();
    let worker = |store: bool| {
        let mut w = crate::generate::Worker::new(HybridStub::new(4096, nv), 8);
        w.ubatch = 4096;
        w.msg_split = true;
        if store {
            let id = crate::prefix_store::identity(std::path::Path::new("/m.gguf"), &format!("{} sched=messages", w.engine.state_id().unwrap()), 4096);
            w.store = Some(PrefixStore::open(&dir, id, 1 << 30, 64).unwrap());
        }
        w
    };
    let sys: Vec<u32> = prompt(600, 20);
    let mut p1 = sys.clone();
    p1.extend(prompt(40, 21));
    let mut p2 = sys.clone();
    p2.extend(prompt(55, 22));
    let gp = crate::generate::GenParams { first_user: Some(600), query_start: true, msg_starts: vec![0, 600], n_predict: 12, ignore_eos: true, ..crate::tests::greedy() };
    let mut a = worker(true);
    a.generate(&tok, &p1, &gp, &mut |_| true).unwrap();
    a.store.as_mut().unwrap().flush();
    assert_eq!(a.store.as_ref().unwrap().n_entries(), 1, "anchor stored at the first user message");
    let mut b = worker(true);
    let rb = b.generate(&tok, &p2, &gp, &mut |_| true).unwrap();
    assert_eq!(rb.timings.cache_n, 600, "restarted process resumes at the anchor");
    assert_eq!(b.engine.loads, 1);
    let mut c = worker(false);
    let rc = c.generate(&tok, &p2, &gp, &mut |_| true).unwrap();
    assert_eq!(rc.timings.cache_n, 0);
    assert_eq!(rb.tokens, rc.tokens, "same greedy output");
    assert_eq!(b.engine.logits_bits(), c.engine.logits_bits(), "same final logits");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Oversized length fields (corrupt or crafted files) are checked BEFORE they size an allocation: each one
/// is a miss + reset (entry deleted), never a huge buffer (an attempted multi-EiB allocation would abort
/// this test process). Covers the store's v2 state length, the token count in the header, and the
/// engine's inner compact length inside an OXEPFX01 entry (which has no outer length).
#[test]
fn flash_prefix_oversized_lengths_miss_without_allocating() {
    let p = prompt(300, 30);
    let mut cold = HybridStub::new(16384, 4096);
    cold.prefill(&p).unwrap();
    // header layout: magic 8 | id len 4 | id | n_tokens 8 | tokens 4n | state len 8 | checksum 8 | state
    let state_len_at = 8 + 4 + ID.len() + 8 + 4 * 256;
    let ntok_at = 8 + 4 + ID.len();
    let patch_u64 = |f: &std::path::Path, at: usize, v: u64| {
        let mut b = std::fs::read(f).unwrap();
        b[at..at + 8].copy_from_slice(&v.to_le_bytes());
        std::fs::write(f, b).unwrap();
    };
    // (1) v2 state length field = 2^62 and = file length + 1
    for huge in [1u64 << 62, u64::MAX, 1 << 40] {
        let dir = scratch_dir("fp-huge-len");
        store_anchor(&dir, ID, &p, 256);
        let f = entries(&dir, "bin").pop().unwrap();
        patch_u64(&f, state_len_at, huge);
        let st = PrefixStore::open(&dir, ID.into(), 1 << 30, 64).unwrap();
        let mut b = HybridStub::new(16384, 4096);
        let err = st.load(&mut b, &p, 256).err().expect("loaded an entry with a bad state length");
        assert!(err.contains("state length"), "{err}");
        assert_eq!(b.loads, 0, "the engine never saw the entry");
        assert!(!f.exists() && st.n_entries() == 0);
        b.reset();
        b.prefill(&p).unwrap();
        assert_eq!(b.logits_bits(), cold.logits_bits());
        let _ = std::fs::remove_dir_all(&dir);
    }
    // (2) token count larger than the file: not indexed at open (no 16 MiB token buffer for a 1 MB file)
    let dir = scratch_dir("fp-huge-ntok");
    store_anchor(&dir, ID, &p, 256);
    let f = entries(&dir, "bin").pop().unwrap();
    patch_u64(&f, ntok_at, (1 << 22) - 1);
    assert_eq!(PrefixStore::open(&dir, ID.into(), 1 << 30, 64).unwrap().n_entries(), 0);
    patch_u64(&f, ntok_at, u64::MAX);
    assert_eq!(PrefixStore::open(&dir, ID.into(), 1 << 30, 64).unwrap().n_entries(), 0);
    let _ = std::fs::remove_dir_all(&dir);
    // (3) OXEPFX01 entry whose engine state says its compact part is 2^62 bytes: the engine checks the length
    // against its layout before allocating (as FlashEngine::load_state does with compact_bytes_at)
    let dir = scratch_dir("fp-huge-inner");
    std::fs::create_dir_all(&dir).unwrap();
    let mut a = HybridStub::new(16384, 4096);
    a.prefill(&p[..256]).unwrap();
    let mut state = a.save_state_vec().unwrap();
    state[..8].copy_from_slice(&(1u64 << 62).to_le_bytes());
    let mut st = PrefixStore::open(&dir, ID.into(), 1 << 30, 64).unwrap();
    st.save_async(p[..256].to_vec(), state.clone());
    st.flush();
    let name = entries(&dir, "bin").pop().unwrap();
    // as a v2 entry (outer length and checksum consistent): the engine's own check refuses it
    let err = st.load(&mut HybridStub::new(16384, 4096), &p, 256).err().expect("v2: loaded a huge compact length");
    assert!(err.contains("compact state size"), "{err}");
    assert!(!name.exists());
    let mut v1 = Vec::new();
    v1.extend_from_slice(b"OXEPFX01");
    v1.extend_from_slice(&(ID.len() as u32).to_le_bytes());
    v1.extend_from_slice(ID.as_bytes());
    v1.extend_from_slice(&256u64.to_le_bytes());
    v1.extend(p[..256].iter().flat_map(|t| t.to_le_bytes()));
    v1.extend_from_slice(&state);
    std::fs::write(&name, v1).unwrap();
    let st = PrefixStore::open(&dir, ID.into(), 1 << 30, 64).unwrap();
    assert_eq!(st.n_entries(), 1);
    let mut b = HybridStub::new(16384, 4096);
    let err = st.load(&mut b, &p, 256).err().expect("loaded a state with a huge compact length");
    assert!(err.contains("compact state size"), "{err}");
    assert!(!name.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// FLASH-PREFIX-2: with a save floor, an entry is built and written only while MemAvailable >= floor + 2 x entry
/// (unknown MemAvailable = no); never while an earlier write still holds its buffer; floor 0 = always.
#[test]
fn flash_prefix_save_floor() {
    let dir = scratch_dir("fp-floor");
    let mut st = PrefixStore::open(&dir, ID.into(), 1 << 30, 64).unwrap();
    const MIB: u64 = 1 << 20;
    st.mem_available = || Some(100 * MIB);
    assert!(st.save_allowed(300 * MIB).is_ok(), "floor 0: always");
    st.save_floor_bytes = 1024 * MIB;
    st.mem_available = || Some(1024 * MIB + 600 * MIB - 1);
    let e = st.save_allowed(300 * MIB).unwrap_err();
    assert!(e.contains("MemAvailable"), "{e}");
    st.mem_available = || Some(1024 * MIB + 600 * MIB);
    assert!(st.save_allowed(300 * MIB).is_ok());
    st.mem_available = || None;
    assert!(st.save_allowed(1).is_err(), "unknown MemAvailable: no save");
    st.mem_available = || Some(u64::MAX);
    assert!(st.save_allowed(u64::MAX).is_ok(), "saturates instead of overflowing");
    let _ = std::fs::remove_dir_all(&dir);
}

/// FLASH-PREFIX-2 through `Worker::generate`: under the floor the anchor is not saved (no state buffer, no file),
/// the request itself is unchanged; with room the same request stores it.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn flash_prefix_worker_skips_save_when_memory_is_short() {
    let Some(m) = crate::tests::model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let nv = tok.n_vocab();
    let mut p = prompt(600, 40);
    p.extend(prompt(40, 41));
    let gp = crate::generate::GenParams { first_user: Some(600), query_start: true, msg_starts: vec![0, 600], n_predict: 8, ignore_eos: true, ..crate::tests::greedy() };
    let run = |avail: fn() -> Option<u64>, tag: &str| {
        let dir = scratch_dir(tag);
        let mut w = crate::generate::Worker::new(HybridStub::new(4096, nv), 8);
        w.ubatch = 4096;
        w.msg_split = true;
        let mut st = PrefixStore::open(&dir, ID.into(), 1 << 30, 64).unwrap();
        st.save_floor_bytes = 1 << 30;
        st.mem_available = avail;
        w.store = Some(st);
        let r = w.generate(&tok, &p, &gp, &mut |_| true).unwrap();
        w.store.as_mut().unwrap().flush();
        let n = entries(&dir, "bin").len();
        let _ = std::fs::remove_dir_all(&dir);
        (r.tokens, n)
    };
    let (short, n_short) = run(|| Some(1 << 30), "fp-floor-short");
    let (room, n_room) = run(|| Some(8 << 30), "fp-floor-room");
    assert_eq!(n_short, 0, "no entry under the floor");
    assert_eq!(n_room, 1, "entry with room");
    assert_eq!(short, room, "the reply does not depend on the save");
}
