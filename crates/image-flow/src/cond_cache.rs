//! On-disk conditioning cache: the text context, RoPE table and sigmas of one (prompt, model, size,
//! steps) combination, so repeated prompts and new seeds run without sd.cpp.
//!
//! Layout: `<root>/<key>/{txt.f32, pe.f32, sigmas.f32, meta.json}`, all little-endian f32 except
//! the JSON. `key` is the SHA-256 (hex) over a length-prefixed encoding of: the format tag, the prompt
//! bytes, the model file name, the model file size, width, height and steps. The prompt text is never
//! written to disk outside the hashed key input; `meta.json` holds only its SHA-256.
//!
//! Encoder provenance: an entry written from a native text encoder carries its encoder identity
//! (`qwen3_te::encoder_identity`: graph version, arithmetic, TE file name and size) in the key and in
//! `meta.json`. Legacy entries (sd.cpp hand-over context) keep the exact legacy key and are marked
//! `"encoder": "sd.cpp-handover"`. The two can never replace or be read for each other, so switching
//! the encoder setting back (rollback) finds the untouched legacy entries again.
//!
//! Corrupt or inconsistent entries are errors, never silently used. The noise is not cached: it
//! comes from the seed (`noise.rs`) or from a hand-over.

use crate::handoff::Handoff;
use crate::latent::grid_for;
use crate::sha256::{hex, sha256};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const FORMAT: &str = "oxi-cond-v1";
/// `meta.json` "encoder" of an entry whose context came from an sd.cpp hand-over.
pub const LEGACY_ENCODER: &str = "sd.cpp-handover";

/// What the key is computed from.
pub struct KeyInputs<'a> {
    pub prompt: &'a [u8],
    /// File name of the model (no directory).
    pub model_name: &'a str,
    /// File size of the model in bytes.
    pub model_size: u64,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// Encoder identity of a native context; `None` = the sd.cpp hand-over context (legacy key).
    pub encoder: Option<&'a str>,
}

/// One cached conditioning entry.
#[derive(Debug, Clone, PartialEq)]
pub struct Conditioning {
    pub n_txt: usize,
    pub n_img: usize,
    /// latent grid (h, w), `h * w == n_img`
    pub grid: (usize, usize),
    pub ctx_dim: usize,
    /// half the head dim: `pe` is `[n_txt + n_img][pe_half][4]`
    pub pe_half: usize,
    /// `[n_txt][ctx_dim]`
    pub txt: Vec<f32>,
    /// `[n_txt + n_img][pe_half][4]`
    pub pe: Vec<f32>,
    /// sigma of each step (`steps` values; the schedule ends at 0)
    pub sigmas: Vec<f32>,
}

fn push_bytes(buf: &mut Vec<u8>, b: &[u8]) {
    buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
    buf.extend_from_slice(b);
}

/// Cache key (hex SHA-256) for the inputs.
pub fn cache_key(k: &KeyInputs) -> String {
    let mut buf = Vec::new();
    push_bytes(&mut buf, FORMAT.as_bytes());
    push_bytes(&mut buf, k.prompt);
    push_bytes(&mut buf, k.model_name.as_bytes());
    buf.extend_from_slice(&k.model_size.to_le_bytes());
    buf.extend_from_slice(&(k.width as u64).to_le_bytes());
    buf.extend_from_slice(&(k.height as u64).to_le_bytes());
    buf.extend_from_slice(&(k.steps as u64).to_le_bytes());
    if let Some(e) = k.encoder {
        // appended only for native entries: legacy keys stay byte-identical
        push_bytes(&mut buf, b"encoder");
        push_bytes(&mut buf, e.as_bytes());
    }
    hex(&sha256(&buf))
}

/// SHA-256 (hex) of the prompt bytes; the only prompt-derived value stored on disk.
pub fn prompt_sha(prompt: &[u8]) -> String {
    hex(&sha256(prompt))
}

/// `$XDG_CACHE_HOME/oxi/cond`, else `$HOME/.cache/oxi/cond`.
pub fn default_root() -> PathBuf {
    let base = match std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        Some(x) => PathBuf::from(x),
        None => match std::env::var_os("HOME").filter(|v| !v.is_empty()) {
            Some(h) => PathBuf::from(h).join(".cache"),
            None => PathBuf::from(".cache"),
        },
    };
    base.join("oxi").join("cond")
}

/// Store a conditioning entry. Returns the entry directory.
pub fn put(root: &Path, k: &KeyInputs, c: &Conditioning) -> Result<PathBuf, String> {
    let (n, nt, ni) = (c.ctx_dim, c.n_txt, c.n_img);
    if c.txt.len() != nt * n {
        return Err(format!("txt: {} values, want {nt} x {n}", c.txt.len()));
    }
    if c.pe.len() != (nt + ni) * c.pe_half * 4 {
        return Err(format!("pe: {} values, want {} x {} x 4", c.pe.len(), nt + ni, c.pe_half));
    }
    if c.grid.0 * c.grid.1 != ni {
        return Err(format!("grid {:?} does not hold {ni} image tokens", c.grid));
    }
    if c.sigmas.len() != k.steps {
        return Err(format!("sigmas: {} values, want {} (one per step)", c.sigmas.len(), k.steps));
    }
    let key = cache_key(k);
    let dir = root.join(&key);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    write_f32(&dir.join("txt.f32"), &c.txt)?;
    write_f32(&dir.join("pe.f32"), &c.pe)?;
    write_f32(&dir.join("sigmas.f32"), &c.sigmas)?;
    let meta = json!({
        "format": FORMAT,
        "key": key,
        "prompt_sha256": prompt_sha(k.prompt),
        "model_name": k.model_name,
        "model_size": k.model_size,
        "width": k.width,
        "height": k.height,
        "steps": k.steps,
        "n_txt": nt,
        "n_img": ni,
        "grid": [c.grid.0, c.grid.1],
        "ctx_dim": n,
        "pe_half": c.pe_half,
        "encoder": k.encoder.unwrap_or(LEGACY_ENCODER),
    });
    // meta.json is written last: an entry without it is incomplete, not valid
    let text = serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("meta.json"), text).map_err(|e| format!("meta.json: {e}"))?;
    Ok(dir)
}

/// Store the conditioning of a sd.cpp hand-over for these inputs. The grid comes from width and height.
pub fn put_from_handoff(root: &Path, k: &KeyInputs, h: &Handoff) -> Result<PathBuf, String> {
    if h.steps != k.steps {
        return Err(format!("hand-over has {} steps, key says {}", h.steps, k.steps));
    }
    let grid = grid_for(k.width, k.height)?;
    let c = Conditioning {
        n_txt: h.n_txt,
        n_img: h.n_img,
        grid,
        ctx_dim: h.ctx_dim,
        pe_half: h.pe_half,
        txt: h.txt.clone(),
        pe: h.pe.clone(),
        sigmas: h.sigmas.clone(),
    };
    put(root, k, &c)
}

/// Store the conditioning of a hand-over that holds fewer model calls than the run it describes
/// (IMG-SHIP: sd-cli is stopped once the first model call's inputs are dumped, so only call 0 exists).
/// The text context and RoPE table do not change over steps; the sigmas of `k.steps` steps come from
/// the ported FLUX.2 schedule and must agree with every dumped sigma within 1e-5, which also checks
/// that the hand-over was made with the same image size.
pub fn put_from_partial_handoff(root: &Path, k: &KeyInputs, h: &Handoff) -> Result<PathBuf, String> {
    if h.steps == 0 || h.steps > k.steps {
        return Err(format!("hand-over has {} model calls, the key wants {} steps", h.steps, k.steps));
    }
    let grid = grid_for(k.width, k.height)?;
    if grid.0 * grid.1 != h.n_img {
        return Err(format!("hand-over has {} image tokens, {}x{} needs {}", h.n_img, k.width, k.height, grid.0 * grid.1));
    }
    let full = crate::schedule::flux2_sigmas(k.steps as u32, h.n_img);
    for (i, &s) in h.sigmas.iter().enumerate() {
        if (s - full[i]).abs() > 1e-5 {
            return Err(format!("dumped sigma {i} = {s} differs from the computed {} (other steps or size?)", full[i]));
        }
    }
    let c = Conditioning {
        n_txt: h.n_txt,
        n_img: h.n_img,
        grid,
        ctx_dim: h.ctx_dim,
        pe_half: h.pe_half,
        txt: h.txt.clone(),
        pe: h.pe.clone(),
        sigmas: full[..k.steps].to_vec(),
    };
    put(root, k, &c)
}

/// Look up an entry. `Ok(None)` when there is no entry for the key; `Err` when an entry is incomplete,
/// inconsistent with the key, or has the wrong file sizes.
pub fn get(root: &Path, k: &KeyInputs) -> Result<Option<Conditioning>, String> {
    let key = cache_key(k);
    let dir = root.join(&key);
    if !dir.exists() {
        return Ok(None);
    }
    let meta_path = dir.join("meta.json");
    let text = std::fs::read_to_string(&meta_path)
        .map_err(|e| format!("{}: incomplete cache entry ({e})", meta_path.display()))?;
    let meta: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", meta_path.display()))?;
    let bad = |what: &str| format!("{}: {what}", meta_path.display());
    let uint = |name: &str| -> Result<usize, String> {
        meta.get(name)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| bad(&format!("missing or non-integer '{name}'")))
    };
    if meta.get("format").and_then(|v| v.as_str()) != Some(FORMAT) {
        return Err(bad("unknown cache format"));
    }
    if meta.get("key").and_then(|v| v.as_str()) != Some(key.as_str()) {
        return Err(bad("key in meta.json does not match the directory name"));
    }
    if meta.get("prompt_sha256").and_then(|v| v.as_str()) != Some(prompt_sha(k.prompt).as_str()) {
        return Err(bad("prompt hash does not match"));
    }
    let enc = meta.get("encoder").and_then(|v| v.as_str()).unwrap_or(LEGACY_ENCODER);
    if enc != k.encoder.unwrap_or(LEGACY_ENCODER) {
        return Err(bad("encoder provenance does not match the key"));
    }
    if uint("steps")? != k.steps || uint("width")? != k.width || uint("height")? != k.height {
        return Err(bad("steps or size do not match the key"));
    }
    let (nt, ni, ctx, pe_half) = (uint("n_txt")?, uint("n_img")?, uint("ctx_dim")?, uint("pe_half")?);
    let grid_v = meta.get("grid").and_then(|v| v.as_array()).ok_or_else(|| bad("missing 'grid'"))?;
    let grid = match (grid_v.first().and_then(|v| v.as_u64()), grid_v.get(1).and_then(|v| v.as_u64())) {
        (Some(h), Some(w)) => (h as usize, w as usize),
        _ => return Err(bad("grid must be [h, w]")),
    };
    if grid.0 * grid.1 != ni {
        return Err(bad("grid does not hold n_img tokens"));
    }
    let want_grid = grid_for(k.width, k.height)?;
    if grid != want_grid {
        return Err(bad("grid does not match the image size"));
    }
    let txt = read_f32(&dir.join("txt.f32"), nt * ctx)?;
    let pe = read_f32(&dir.join("pe.f32"), (nt + ni) * pe_half * 4)?;
    let sigmas = read_f32(&dir.join("sigmas.f32"), k.steps)?;
    Ok(Some(Conditioning { n_txt: nt, n_img: ni, grid, ctx_dim: ctx, pe_half, txt, pe, sigmas }))
}

fn write_f32(path: &Path, v: &[f32]) -> Result<(), String> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn read_f32(path: &Path, count: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != count * 4 {
        return Err(format!("{}: {} bytes, want {} ({count} f32)", path.display(), bytes.len(), count * 4));
    }
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_namespace_keeps_legacy_keys_and_never_crosses() {
        let root = tmp("enc");
        let legacy = inputs(b"a cat", "dit.gguf", 7, 32, 48, 4);
        // the legacy key must not change (existing caches stay valid; rollback finds them)
        let mut buf = Vec::new();
        push_bytes(&mut buf, FORMAT.as_bytes());
        push_bytes(&mut buf, b"a cat");
        push_bytes(&mut buf, b"dit.gguf");
        for v in [7u64, 32, 48, 4] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(cache_key(&legacy), hex(&sha256(&buf)));
        let native = KeyInputs { encoder: Some("qwen3-te/1;cpu-ggml;te.gguf;9"), ..inputs(b"a cat", "dit.gguf", 7, 32, 48, 4) };
        let other = KeyInputs { encoder: Some("qwen3-te/1;npu-fp16;te.gguf;9"), ..inputs(b"a cat", "dit.gguf", 7, 32, 48, 4) };
        assert_ne!(cache_key(&legacy), cache_key(&native));
        assert_ne!(cache_key(&native), cache_key(&other));
        let mut a = sample();
        put(&root, &legacy, &a).unwrap();
        a.txt.iter_mut().for_each(|v| *v += 1.0);
        put(&root, &native, &a).unwrap();
        assert_eq!(get(&root, &legacy).unwrap().unwrap(), sample(), "native write left the legacy entry alone");
        assert_eq!(get(&root, &native).unwrap().unwrap(), a);
        assert!(get(&root, &other).unwrap().is_none());
        // a legacy-keyed entry whose meta claims another encoder is refused, never used
        let dir = root.join(cache_key(&legacy));
        let m = std::fs::read_to_string(dir.join("meta.json")).unwrap().replace(LEGACY_ENCODER, "qwen3-te/1;cpu-ggml;te.gguf;9");
        std::fs::write(dir.join("meta.json"), m).unwrap();
        assert!(get(&root, &legacy).unwrap_err().contains("encoder provenance"));
        let _ = std::fs::remove_dir_all(&root);
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("image-flow-cond-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn inputs<'a>(prompt: &'a [u8], name: &'a str, size: u64, w: usize, h: usize, steps: usize) -> KeyInputs<'a> {
        KeyInputs { prompt, model_name: name, model_size: size, width: w, height: h, steps, encoder: None }
    }

    /// 32 x 48 px image: grid (3, 2), 6 image tokens; 3 text tokens of width 4; pe_half 2.
    fn sample() -> Conditioning {
        Conditioning {
            n_txt: 3,
            n_img: 6,
            grid: (3, 2),
            ctx_dim: 4,
            pe_half: 2,
            txt: (0..12).map(|i| i as f32 * 0.25 - 1.0).collect(),
            pe: (0..9 * 2 * 4).map(|i| (i as f32).sin()).collect(),
            sigmas: vec![1.0, 0.5, 0.25, 0.0],
        }
    }

    #[test]
    fn key_depends_on_every_input_and_is_stable() {
        let base = cache_key(&inputs(b"a cat", "klein.gguf", 100, 32, 48, 4));
        assert_eq!(base.len(), 64);
        assert_eq!(base, cache_key(&inputs(b"a cat", "klein.gguf", 100, 32, 48, 4)));
        assert_ne!(base, cache_key(&inputs(b"a dog", "klein.gguf", 100, 32, 48, 4)));
        assert_ne!(base, cache_key(&inputs(b"a cat", "other.gguf", 100, 32, 48, 4)));
        assert_ne!(base, cache_key(&inputs(b"a cat", "klein.gguf", 101, 32, 48, 4)));
        assert_ne!(base, cache_key(&inputs(b"a cat", "klein.gguf", 100, 48, 32, 4)));
        assert_ne!(base, cache_key(&inputs(b"a cat", "klein.gguf", 100, 32, 48, 5)));
        assert_ne!(base, cache_key(&inputs(b"a cat", "klein.gguf", 100, 32, 64, 4)));
    }

    #[test]
    fn put_then_get_round_trips_exact_bits() {
        let root = tmp("rt");
        let k = inputs(b"a red fox", "klein.gguf", 4096, 32, 48, 4);
        let c = sample();
        let dir = put(&root, &k, &c).unwrap();
        assert!(dir.join("meta.json").exists());
        let got = get(&root, &k).unwrap().expect("entry");
        assert_eq!(got, c);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got.txt), bits(&c.txt));
        assert_eq!(bits(&got.pe), bits(&c.pe));
        assert_eq!(bits(&got.sigmas), bits(&c.sigmas));
        // the prompt text is not on disk
        let meta = std::fs::read_to_string(dir.join("meta.json")).unwrap();
        assert!(!meta.contains("red fox"), "prompt text leaked into meta.json");
        assert!(meta.contains(&prompt_sha(b"a red fox")));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn missing_entry_is_none() {
        let root = tmp("missing");
        let k = inputs(b"nothing", "klein.gguf", 1, 32, 48, 4);
        assert!(get(&root, &k).unwrap().is_none());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn corrupt_entries_are_errors() {
        let root = tmp("corrupt");
        let k = inputs(b"p", "m.gguf", 9, 32, 48, 4);
        let dir = put(&root, &k, &sample()).unwrap();

        // truncated txt
        let txt = dir.join("txt.f32");
        let full = std::fs::read(&txt).unwrap();
        std::fs::write(&txt, &full[..full.len() - 4]).unwrap();
        let e = get(&root, &k).err().expect("truncated txt must fail");
        assert!(e.contains("txt.f32"), "{e}");
        std::fs::write(&txt, &full).unwrap();
        assert!(get(&root, &k).unwrap().is_some());

        // missing meta.json: incomplete entry
        std::fs::remove_file(dir.join("meta.json")).unwrap();
        assert!(get(&root, &k).err().expect("incomplete").contains("incomplete"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn meta_mismatch_is_an_error() {
        let root = tmp("meta");
        let k = inputs(b"p", "m.gguf", 9, 32, 48, 4);
        let dir = put(&root, &k, &sample()).unwrap();
        let text = std::fs::read_to_string(dir.join("meta.json")).unwrap();
        std::fs::write(dir.join("meta.json"), text.replace("\"steps\": 4", "\"steps\": 7")).unwrap();
        assert!(get(&root, &k).err().expect("steps mismatch").contains("steps"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn put_rejects_inconsistent_shapes() {
        let root = tmp("shape");
        let k = inputs(b"p", "m.gguf", 9, 32, 48, 4);
        let mut c = sample();
        c.txt.pop();
        assert!(put(&root, &k, &c).is_err());
        let mut c = sample();
        c.grid = (2, 2);
        assert!(put(&root, &k, &c).is_err());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn put_from_handoff_stores_the_dump_conditioning() {
        use crate::handoff::Handoff;
        use std::path::PathBuf;
        let root = tmp("ho");
        let k = inputs(b"hand", "m.gguf", 9, 32, 48, 2);
        let h = Handoff {
            dir: PathBuf::from("/nonexistent"),
            n_img: 6,
            n_txt: 3,
            in_ch: 8,
            ctx_dim: 4,
            out_ch: 8,
            steps: 2,
            sigmas: vec![1.0, 0.5],
            sigma_end: 0.0,
            img0: vec![0.0; 48],
            txt: (0..12).map(|i| i as f32).collect(),
            pe: vec![0.5; 9 * 2 * 4],
            pe_half: 2,
            img_steps: vec![],
            out_steps: vec![None, None],
            single_in: [vec![None, None], vec![None, None]],
            single_hidden: None,
            vae_z: None,
            vae_image: None,
        };
        put_from_handoff(&root, &k, &h).unwrap();
        let got = get(&root, &k).unwrap().unwrap();
        assert_eq!(got.txt, h.txt);
        assert_eq!(got.sigmas, vec![1.0, 0.5]);
        assert_eq!(got.grid, (3, 2));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn default_root_ends_in_oxi_cond() {
        assert!(default_root().ends_with("oxi/cond"));
    }

    fn partial_handoff(n_img: usize, sigma0: f32) -> Handoff {
        Handoff {
            dir: PathBuf::from("/nonexistent"),
            n_img,
            n_txt: 3,
            in_ch: 8,
            ctx_dim: 4,
            out_ch: 8,
            steps: 1,
            sigmas: vec![sigma0],
            sigma_end: 0.0,
            img0: vec![0.0; n_img * 8],
            txt: (0..12).map(|i| i as f32 * 0.5).collect(),
            pe: vec![0.25; (3 + n_img) * 2 * 4],
            pe_half: 2,
            img_steps: vec![],
            out_steps: vec![None],
            single_in: [vec![None], vec![None]],
            single_hidden: None,
            vae_z: None,
            vae_image: None,
        }
    }

    #[test]
    fn partial_handoff_stores_computed_sigmas_for_all_steps() {
        let root = tmp("partial");
        let k = inputs(b"part", "m.gguf", 9, 32, 48, 4);
        let h = partial_handoff(6, 1.0);
        put_from_partial_handoff(&root, &k, &h).unwrap();
        let got = get(&root, &k).unwrap().unwrap();
        let want = crate::schedule::flux2_sigmas(4, 6);
        assert_eq!(got.sigmas, want[..4].to_vec());
        assert_eq!(want[4], 0.0, "stored sigmas end at 0, as generate assumes");
        assert_eq!(got.txt, h.txt);
        assert_eq!(got.pe, h.pe);
        assert_eq!(got.grid, (3, 2));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn partial_handoff_rejects_wrong_size_sigma_or_too_many_calls() {
        let root = tmp("partial-bad");
        let k = inputs(b"part", "m.gguf", 9, 32, 48, 4);
        // 8 tokens do not fit a 32x48 image (3x2 grid)
        assert!(put_from_partial_handoff(&root, &k, &partial_handoff(8, 1.0)).unwrap_err().contains("image tokens"));
        // a first sigma that is not the schedule's
        assert!(put_from_partial_handoff(&root, &k, &partial_handoff(6, 0.9)).unwrap_err().contains("sigma"));
        // more calls than steps
        let mut h = partial_handoff(6, 1.0);
        h.steps = 5;
        h.sigmas = vec![1.0; 5];
        assert!(put_from_partial_handoff(&root, &k, &h).is_err());
        assert!(get(&root, &k).unwrap().is_none(), "nothing stored on error");
        std::fs::remove_dir_all(&root).ok();
    }
}
