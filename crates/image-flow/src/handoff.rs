//! Reader for a hand-over directory written by the patched sd.cpp (`SDPROBE` dumps).
//!
//! File names are `<tag>.call<n>.<type>.<a>x<b>x<c>x<d>.bin`; `ne[0]` (the first extent) is the
//! fastest-varying index; data is little-endian `f32` or `f16`. Per-step tags have one file per
//! call (sampling step), the call index is `n`.
//!
//! Tags read here (see `forward_orig` in `model/diffusion/flux.hpp`):
//! - `dump.flux.img`        `[in_ch, n_img, 1, 1]`, latent tokens entering the model (call n = step n)
//! - `dump.flux.txt`        `[ctx, n_txt, 1, 1]`, context before `txt_in`
//! - `dump.flux.timesteps`  `[1]` per call, or `[steps]` in one file: the sigma of each step
//! - `dump.pe`              `[2, 2, head_dim/2, n_txt + n_img]`
//! - `dump.flux.out`        `[out_ch, n_img, 1, 1]`, model output of the step (optional)
//! - `dump.single.0.in`, `dump.single.1.in`  `[hidden, n_txt + n_img, 1, 1]` (optional)
//! - `dump.vae.z`           `[w, h, 128, 1]`, latent given to the decoder (optional)
//! - `dump.vae.image`       `[W, H, 3, 1]` (optional)

use image_dit::dump::DumpTensor;
use int8_gemm::f16_to_f32;
use std::path::{Path, PathBuf};

/// klein: latent channels into `img_in`, context width into `txt_in`, output channels.
pub const KLEIN_IN_CH: usize = 128;
pub const KLEIN_CTX: usize = 7680;
pub const KLEIN_OUT_CH: usize = 128;

pub struct Handoff {
    pub dir: PathBuf,
    pub n_img: usize,
    pub n_txt: usize,
    pub in_ch: usize,
    pub ctx_dim: usize,
    pub out_ch: usize,
    /// Number of sampling steps (= number of model calls).
    pub steps: usize,
    /// Model timestep (sigma) of each step, length `steps`.
    pub sigmas: Vec<f32>,
    /// Sigma after the last step (the schedule ends at 0).
    pub sigma_end: f32,
    /// `[n_img][in_ch]`: call 0 of `dump.flux.img` (the initial noise).
    pub img0: Vec<f32>,
    /// `[n_txt][ctx]`: call 0 of `dump.flux.txt`.
    pub txt: Vec<f32>,
    /// `[n][head_dim/2][4]` flattened from `[2,2,head_dim/2,n]`, n = n_txt + n_img.
    pub pe: Vec<f32>,
    pub pe_half: usize,
    /// Per step: `[n_img][in_ch]` entering the model.
    pub img_steps: Vec<Vec<f32>>,
    /// Per step: `[n_img][out_ch]` model output, when dumped.
    pub out_steps: Vec<Option<Vec<f32>>>,
    /// Per step, for single blocks 0 and 1: `[n][hidden]` entering the block, when dumped.
    pub single_in: [Vec<Option<Vec<f32>>>; 2],
    /// Hidden size of the single-block dumps, when present.
    pub single_hidden: Option<usize>,
    /// Latent given to the VAE decoder (the last call), `ne [w, h, 128, 1]`.
    pub vae_z: Option<DumpTensor>,
    /// Decoded reference image, `ne [W, H, 3, 1]`.
    pub vae_image: Option<DumpTensor>,
}

impl Handoff {
    /// Load a klein hand-over (shapes: in 128, context 7680, out 128).
    pub fn load(dir: &Path) -> Result<Handoff, String> {
        Handoff::load_with(dir, KLEIN_IN_CH, KLEIN_CTX, KLEIN_OUT_CH)
    }

    /// Load with explicit model widths (tests use tiny ones).
    pub fn load_with(dir: &Path, in_ch: usize, ctx_dim: usize, out_ch: usize) -> Result<Handoff, String> {
        let img_calls = list_calls(dir, "dump.flux.img")?;
        if img_calls.is_empty() {
            return Err(format!("{}: no dump.flux.img.call<n> files", dir.display()));
        }
        let steps = img_calls.len();
        if img_calls != (0..steps).collect::<Vec<_>>() {
            return Err(format!("dump.flux.img calls are not 0..{steps}: {img_calls:?}"));
        }

        // img per step
        let first = req_call(dir, "dump.flux.img", 0)?;
        let n_img = first.ne[1];
        expect_ne(&first, "dump.flux.img", [in_ch, n_img, 1, 1])?;
        if n_img == 0 {
            return Err("dump.flux.img has no tokens".into());
        }
        let mut img_steps = Vec::with_capacity(steps);
        for c in 0..steps {
            let t = req_call(dir, "dump.flux.img", c)?;
            expect_ne(&t, &format!("dump.flux.img call {c}"), [in_ch, n_img, 1, 1])?;
            img_steps.push(t.data);
        }
        let img0 = img_steps[0].clone();

        // txt (call 0; the context does not change over steps)
        let txt_t = req_call(dir, "dump.flux.txt", 0)?;
        let n_txt = txt_t.ne[1];
        expect_ne(&txt_t, "dump.flux.txt", [ctx_dim, n_txt, 1, 1])?;
        let txt = txt_t.data;

        // timesteps: one file per call with one element, or one file with `steps` elements
        let ts_calls = list_calls(dir, "dump.flux.timesteps")?;
        let sigmas = if ts_calls.len() == steps && ts_calls == (0..steps).collect::<Vec<_>>() {
            let mut v = Vec::with_capacity(steps);
            for c in 0..steps {
                let t = req_call(dir, "dump.flux.timesteps", c)?;
                if t.data.len() != 1 {
                    return Err(format!("dump.flux.timesteps call {c}: {} elements, want 1", t.data.len()));
                }
                v.push(t.data[0]);
            }
            v
        } else if ts_calls == vec![0] {
            let t = req_call(dir, "dump.flux.timesteps", 0)?;
            if t.data.len() != steps {
                return Err(format!(
                    "dump.flux.timesteps: {} elements in one file, want {steps} (one per step)",
                    t.data.len()
                ));
            }
            t.data
        } else {
            return Err(format!(
                "dump.flux.timesteps: calls {ts_calls:?}, want one file per step (0..{steps}) or a single file"
            ));
        };

        // pe (call 0)
        let pe_t = req_call(dir, "dump.pe", 0)?;
        let n = n_txt + n_img;
        if pe_t.ne[0] != 2 || pe_t.ne[1] != 2 || pe_t.ne[3] != n || pe_t.ne[2] == 0 {
            return Err(format!(
                "dump.pe: ne {:?}, want [2, 2, head_dim/2, {n}] (n_txt {n_txt} + n_img {n_img})",
                pe_t.ne
            ));
        }
        let pe_half = pe_t.ne[2];
        let pe = pe_t.data;

        // optional per-step outputs
        let out_calls = list_calls(dir, "dump.flux.out")?;
        let mut out_steps: Vec<Option<Vec<f32>>> = vec![None; steps];
        for c in out_calls {
            if c >= steps {
                return Err(format!("dump.flux.out call {c} beyond {steps} steps"));
            }
            let t = req_call(dir, "dump.flux.out", c)?;
            expect_ne(&t, &format!("dump.flux.out call {c}"), [out_ch, n_img, 1, 1])?;
            out_steps[c] = Some(t.data);
        }

        // optional single-block inputs
        let mut single_hidden = None;
        let mut single_in: [Vec<Option<Vec<f32>>>; 2] = [vec![None; steps], vec![None; steps]];
        for (b, tag) in ["dump.single.0.in", "dump.single.1.in"].iter().enumerate() {
            for c in list_calls(dir, tag)? {
                if c >= steps {
                    return Err(format!("{tag} call {c} beyond {steps} steps"));
                }
                let t = req_call(dir, tag, c)?;
                if t.ne[2] != 1 || t.ne[3] != 1 || t.ne[1] != n {
                    return Err(format!("{tag}: ne {:?}, want [hidden, {n}, 1, 1]", t.ne));
                }
                let hidden = t.ne[0];
                match single_hidden {
                    None => single_hidden = Some(hidden),
                    Some(h) if h != hidden => {
                        return Err(format!("{tag}: hidden {hidden} differs from earlier dumps ({h})"))
                    }
                    _ => {}
                }
                single_in[b][c] = Some(t.data);
            }
        }

        // optional VAE tensors: the last call is the one that matters
        let vae_z = match list_calls(dir, "dump.vae.z")?.into_iter().max() {
            Some(c) => {
                let t = req_call(dir, "dump.vae.z", c)?;
                if t.ne[2] != in_ch || t.ne[3] != 1 || t.ne[0] == 0 || t.ne[1] == 0 {
                    return Err(format!("dump.vae.z: ne {:?}, want [w, h, {in_ch}, 1]", t.ne));
                }
                Some(t)
            }
            None => None,
        };
        let vae_image = match list_calls(dir, "dump.vae.image")?.into_iter().max() {
            Some(c) => Some(req_call(dir, "dump.vae.image", c)?),
            None => None,
        };

        Ok(Handoff {
            dir: dir.to_path_buf(),
            n_img,
            n_txt,
            in_ch,
            ctx_dim,
            out_ch,
            steps,
            sigmas,
            sigma_end: 0.0,
            img0,
            txt,
            pe,
            pe_half,
            img_steps,
            out_steps,
            single_in,
            single_hidden,
            vae_z,
            vae_image,
        })
    }

    /// Model input tokens of step `i`.
    pub fn img_at(&self, i: usize) -> &[f32] {
        &self.img_steps[i]
    }

    /// Model output dumped at step `i`, if any.
    pub fn out_at(&self, i: usize) -> Option<&[f32]> {
        self.out_steps.get(i).and_then(|o| o.as_deref())
    }

    /// Input of single block `block` (0 or 1) at step `i`, if dumped.
    pub fn single_in_at(&self, block: usize, i: usize) -> Option<&[f32]> {
        self.single_in.get(block).and_then(|v| v.get(i)).and_then(|o| o.as_deref())
    }

    /// Sigma of step `i` and the sigma after it (0 after the last step).
    pub fn sigma_pair(&self, i: usize) -> (f32, f32) {
        let next = if i + 1 < self.steps { self.sigmas[i + 1] } else { self.sigma_end };
        (self.sigmas[i], next)
    }

    /// Latent grid `(h, w)` from `dump.vae.z`, if present.
    pub fn vae_grid(&self) -> Option<(usize, usize)> {
        self.vae_z.as_ref().map(|z| (z.ne[1], z.ne[0]))
    }
}

/// Call indices present for `tag` (any `<tag>.call<n>.` name), sorted.
pub fn list_calls(dir: &Path, tag: &str) -> Result<Vec<usize>, String> {
    let prefix = format!("{tag}.call");
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut calls = Vec::new();
    for ent in rd {
        let ent = ent.map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = ent.file_name().to_string_lossy().into_owned();
        if let Some(rest) = name.strip_prefix(&prefix) {
            if let Some(dot) = rest.find('.') {
                if let Ok(n) = rest[..dot].parse::<usize>() {
                    calls.push(n);
                }
            }
        }
    }
    calls.sort_unstable();
    calls.dedup();
    Ok(calls)
}

/// The dump of `tag` at `call`, or `None` when no file exists. More than one file is an error.
pub fn read_tag_call(dir: &Path, tag: &str, call: usize) -> Result<Option<DumpTensor>, String> {
    let prefix = format!("{tag}.call{call}.");
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut hits: Vec<String> = Vec::new();
    for ent in rd {
        let ent = ent.map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = ent.file_name().to_string_lossy().into_owned();
        if name.starts_with(&prefix) {
            hits.push(name);
        }
    }
    hits.sort();
    match hits.len() {
        0 => return Ok(None),
        1 => {}
        n => return Err(format!("{n} dump files for {tag} call {call} in {}: {hits:?}", dir.display())),
    }
    let name = &hits[0];
    let bad = || format!("bad dump file name '{name}' (want <tag>.call<n>.<type>.<a>x<b>x<c>x<d>.bin)");
    let body = name[prefix.len()..].strip_suffix(".bin").ok_or_else(bad)?;
    let parts: Vec<&str> = body.split('.').collect();
    if parts.len() != 2 {
        return Err(bad());
    }
    let ty = parts[0];
    let dims: Vec<usize> = parts[1]
        .split('x')
        .map(|s| s.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| bad())?;
    if dims.len() != 4 {
        return Err(bad());
    }
    let ne = [dims[0], dims[1], dims[2], dims[3]];
    let count: usize = ne.iter().product();
    let esize = match ty {
        "f32" => 4usize,
        "f16" => 2usize,
        other => return Err(format!("{name}: unsupported element type '{other}'")),
    };
    let path = dir.join(name);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != count * esize {
        return Err(format!(
            "{name}: file has {} bytes, ne {ne:?} as {ty} needs {}",
            bytes.len(),
            count * esize
        ));
    }
    let data: Vec<f32> = if ty == "f32" {
        bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    } else {
        bytes.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
    };
    Ok(Some(DumpTensor { ne, data }))
}

fn req_call(dir: &Path, tag: &str, call: usize) -> Result<DumpTensor, String> {
    read_tag_call(dir, tag, call)?.ok_or_else(|| format!("{}: missing {tag} call {call}", dir.display()))
}

fn expect_ne(t: &DumpTensor, what: &str, want: [usize; 4]) -> Result<(), String> {
    if t.ne != want {
        return Err(format!("{what}: ne {:?}, want {want:?}", t.ne));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("image-flow-handoff-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write(dir: &Path, tag: &str, call: usize, ne: [usize; 4], data: &[f32]) {
        assert_eq!(ne.iter().product::<usize>(), data.len());
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let name = format!("{tag}.call{call}.f32.{}x{}x{}x{}.bin", ne[0], ne[1], ne[2], ne[3]);
        std::fs::write(dir.join(name), bytes).unwrap();
    }

    fn vals(n: usize, base: f32) -> Vec<f32> {
        (0..n).map(|i| base + i as f32 * 0.01).collect()
    }

    // tiny: in 8, ctx 24, out 8, n_img 7, n_txt 5, head_dim 16 (pe half 8), hidden 32, 2 steps
    fn write_valid(dir: &Path) {
        let (ic, cx, oc, ni, nt, half, hid, steps) = (8usize, 24usize, 8usize, 7usize, 5usize, 8usize, 32usize, 2usize);
        for c in 0..steps {
            write(dir, "dump.flux.img", c, [ic, ni, 1, 1], &vals(ic * ni, c as f32));
            write(dir, "dump.flux.timesteps", c, [1, 1, 1, 1], &[1.0 - c as f32 * 0.5]);
            write(dir, "dump.flux.out", c, [oc, ni, 1, 1], &vals(oc * ni, 5.0));
        }
        write(dir, "dump.flux.txt", 0, [cx, nt, 1, 1], &vals(cx * nt, 2.0));
        write(dir, "dump.pe", 0, [2, 2, half, nt + ni], &vals(4 * half * (nt + ni), 0.5));
        write(dir, "dump.single.0.in", 0, [hid, nt + ni, 1, 1], &vals(hid * (nt + ni), 1.0));
        write(dir, "dump.single.0.in", 1, [hid, nt + ni, 1, 1], &vals(hid * (nt + ni), 1.5));
        // dump.single.1.in deliberately missing (optional)
        write(dir, "dump.vae.z", 1, [4, 2, ic, 1], &vals(4 * 2 * ic, 3.0));
    }

    #[test]
    fn round_trip_with_optional_dumps_missing() {
        let dir = tmpdir("ok");
        write_valid(&dir);
        let h = Handoff::load_with(&dir, 8, 24, 8).expect("load");
        assert_eq!((h.n_img, h.n_txt, h.steps), (7, 5, 2));
        assert_eq!(h.sigmas, vec![1.0, 0.5]);
        assert_eq!(h.sigma_pair(0), (1.0, 0.5));
        assert_eq!(h.sigma_pair(1), (0.5, 0.0));
        assert_eq!(h.img0, vals(8 * 7, 0.0));
        assert_eq!(h.img_at(1), &vals(8 * 7, 1.0)[..]);
        assert_eq!(h.txt.len(), 24 * 5);
        assert_eq!(h.pe_half, 8);
        assert_eq!(h.pe.len(), 4 * 8 * 12);
        assert_eq!(h.out_at(1).map(|o| o.len()), Some(8 * 7));
        assert!(h.single_in_at(0, 0).is_some());
        assert!(h.single_in_at(0, 1).is_some());
        assert!(h.single_in_at(1, 0).is_none(), "missing optional tap must be None");
        assert_eq!(h.single_hidden, Some(32));
        assert_eq!(h.vae_grid(), Some((2, 4)));
        assert!(h.vae_image.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_optional_outputs_are_none_not_errors() {
        let dir = tmpdir("minimal");
        let (ic, cx, ni, nt) = (8usize, 24usize, 7usize, 5usize);
        write(&dir, "dump.flux.img", 0, [ic, ni, 1, 1], &vals(ic * ni, 0.0));
        write(&dir, "dump.flux.timesteps", 0, [1, 1, 1, 1], &[1.0]);
        write(&dir, "dump.flux.txt", 0, [cx, nt, 1, 1], &vals(cx * nt, 0.0));
        write(&dir, "dump.pe", 0, [2, 2, 8, nt + ni], &vals(32 * (nt + ni), 0.0));
        let h = Handoff::load_with(&dir, 8, 24, 8).expect("load");
        assert_eq!(h.steps, 1);
        assert!(h.out_at(0).is_none());
        assert!(h.vae_z.is_none());
        assert_eq!(h.sigma_pair(0), (1.0, 0.0));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bad_shapes_and_sizes_are_clear_errors() {
        // wrong token count on call 1
        let dir = tmpdir("badshape");
        write_valid(&dir);
        std::fs::remove_file(dir.join("dump.flux.img.call1.f32.8x7x1x1.bin")).unwrap();
        write(&dir, "dump.flux.img", 1, [8, 6, 1, 1], &vals(48, 0.0));
        let e = Handoff::load_with(&dir, 8, 24, 8).err().expect("must fail");
        assert!(e.contains("dump.flux.img call 1") && e.contains("want"), "{e}");
        std::fs::remove_dir_all(&dir).ok();

        // pe with the wrong token count
        let dir = tmpdir("badpe");
        write_valid(&dir);
        write(&dir, "dump.pe", 0, [2, 2, 8, 10], &vals(320, 0.0));
        let e = Handoff::load_with(&dir, 8, 24, 8).err().expect("must fail");
        assert!(e.contains("dump.pe"), "{e}");
        std::fs::remove_dir_all(&dir).ok();

        // file whose byte count does not match its name
        let dir = tmpdir("badsize");
        write_valid(&dir);
        let name = format!("dump.flux.txt.call0.f32.24x5x1x1.bin");
        std::fs::write(dir.join(name), vec![0u8; 10]).unwrap();
        let e = Handoff::load_with(&dir, 8, 24, 8).err().expect("must fail");
        assert!(e.contains("bytes"), "{e}");
        std::fs::remove_dir_all(&dir).ok();

        // gap in the img call indices
        let dir = tmpdir("gap");
        write_valid(&dir);
        std::fs::remove_file(dir.join("dump.flux.img.call0.f32.8x7x1x1.bin")).unwrap();
        let e = Handoff::load_with(&dir, 8, 24, 8).err().expect("must fail");
        assert!(e.contains("not 0..") || e.contains("missing"), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn f16_dump_is_decoded() {
        let dir = tmpdir("f16");
        // 1.0 = 0x3c00, 2.0 = 0x4000
        std::fs::write(dir.join("dump.flux.img.call0.f16.2x1x1x1.bin"), [0x00u8, 0x3c, 0x00, 0x40]).unwrap();
        let t = read_tag_call(&dir, "dump.flux.img", 0).unwrap().unwrap();
        assert_eq!(t.ne, [2, 1, 1, 1]);
        assert_eq!(t.data, vec![1.0, 2.0]);
        assert!(read_tag_call(&dir, "dump.flux.img", 1).unwrap().is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}
