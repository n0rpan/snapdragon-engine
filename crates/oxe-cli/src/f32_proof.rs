//! Private, bounded O62 proof. Export on CPU first, then use the identical
//! positions/history in a separate accelerator process. Never a server setting.
use super::*;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

fn write_new(path: &Path, bytes: &[u8]) {
    let mut f = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)
        .unwrap_or_else(|e| die(&format!("{}: {e}", path.display())));
    f.write_all(bytes).unwrap();
    f.sync_all().unwrap();
}
fn bytes(rows: &[f32]) -> Vec<u8> {
    rows.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn read_rows(dir: &Path, name: &str, count: usize, expected: &str) -> Vec<f32> {
    let mut f = File::open(dir.join(name)).unwrap_or_else(|e| die(&e.to_string()));
    if f.metadata().unwrap().len() != count.checked_mul(4).unwrap() as u64 {
        die("f32 proof row-file length mismatch");
    }
    let mut b = Vec::new();
    f.read_to_end(&mut b).unwrap();
    if format!("{:016x}", model::weights::hash_bytes(&b)) != expected {
        die("f32 proof row-file checksum mismatch");
    }
    let v: Vec<f32> = b
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if v.iter().any(|x| !x.is_finite()) {
        die("nonfinite f32 proof row");
    }
    v
}
fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}
fn verified_source(a: &Args, expected: &str) -> (gguf::Gguf, String) {
    let g = gguf::Gguf::open(a.model()).unwrap_or_else(|e| die(&e.to_string()));
    let dir =
        model::weights::cache_dir(a.load_opts().cache_dir.as_deref()).unwrap_or_else(|e| die(&e));
    let sha = model::weights::model_sha256_opt(&g.path, &dir, false).unwrap_or_else(|e| die(&e));
    if sha != expected {
        die("f32 proof original source / CPU identity mismatch");
    }
    for t in &g.header.tensors {
        g.release_range(t, 0, t.bytes() as usize);
        g.drop_cache(t);
    }
    (g, sha)
}

pub(super) fn export(a: &Args) {
    if a.get("npu-prefill").is_some() {
        die("CPU reference export must not open an NPU");
    }
    let p = if a.get("tokens").is_some() || a.get("ref").is_some() {
        prompt(a)
    } else {
        (0..a.num("prompt-len", 256) as u32)
            .map(|i| i.wrapping_mul(2654435761) % 150000 + 1000)
            .collect()
    };
    let fixed_teacher = a.get("teacher-ids").map(|v| {
        v.split(',')
            .map(|x| {
                x.parse::<u32>()
                    .unwrap_or_else(|_| die("invalid --teacher-ids"))
            })
            .collect::<Vec<_>>()
    });
    let n = a.num("n", fixed_teacher.as_ref().map_or(16, |x| x.len()));
    if fixed_teacher.as_ref().is_some_and(|x| x.len() != n) {
        die("teacher length / n mismatch");
    }
    let split_last = a.get("split-last") == Some("1");
    let n_ctx = a.num("ctx", 2048);
    let ub = a.num("ubatch", 256);
    if !(33..=512).contains(&p.len())
        || !(1..=256).contains(&n)
        || p.len() + n >= n_ctx
        || n_ctx > 4096
        || ub == 0
    {
        die("f32 export needs33..512 prompt,1..256 rows,ctx<=4096");
    }
    let dir = Path::new(
        a.get("out")
            .unwrap_or_else(|| die("f32-export --out NEW_DIRECTORY required")),
    );
    std::fs::create_dir(dir).unwrap_or_else(|e| die(&e.to_string()));
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    let s = setup(a, n_ctx);
    if p.iter().any(|&t| t as usize >= s.loaded.plan.desc.n_vocab) {
        die("prompt id outside vocabulary");
    }
    let (g, sha) = verified_source(a, &s.loaded.ws.info.model_sha256);
    let cfg = EngineConfig {
        n_ctx,
        logits: true,
        trace: false,
        profile: false,
    };
    let cpu = Engine::new(&s.loaded.plan, &s.loaded.ws, cfg.clone()).unwrap_or_else(|e| die(&e));
    let mut reference = Engine::new(&s.loaded.plan, &s.loaded.ws, cfg).unwrap_or_else(|e| die(&e));
    reference
        .enable_f32_reference(g, &sha)
        .unwrap_or_else(|e| die(&e));
    let t = Instant::now();
    if fixed_teacher
        .as_ref()
        .is_some_and(|x| x.iter().any(|&t| t as usize >= s.loaded.plan.desc.n_vocab))
    {
        die("teacher id outside vocabulary");
    }
    let mut next = if split_last {
        cpu.prefill(&s.pool, &p[..p.len() - 1], ub).unwrap();
        cpu.step(&s.pool, *p.last().unwrap())
    } else {
        cpu.prefill(&s.pool, &p, ub).unwrap()
    };
    let mut teacher = Vec::new();
    let mut cpu_rows = Vec::new();
    for i in 0..n {
        cpu_rows.extend_from_slice(cpu.logits());
        teacher.push(fixed_teacher.as_ref().map_or(next, |x| x[i]));
        if i + 1 < n {
            next = cpu.step(&s.pool, teacher[i]);
        }
    }
    let cpu_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    if split_last {
        reference.prefill(&s.pool, &p[..p.len() - 1], ub).unwrap();
        reference.step(&s.pool, *p.last().unwrap());
    } else {
        reference.prefill(&s.pool, &p, ub).unwrap();
    }
    let mut ref_rows = Vec::new();
    let mut per_row = Vec::new();
    for i in 0..n {
        let r = reference.logits();
        if r.iter().any(|x| !x.is_finite()) {
            die("nonfinite f32 reference");
        }
        ref_rows.extend_from_slice(r);
        let c = &cpu_rows[i * r.len()..(i + 1) * r.len()];
        let (first, second) = top2(r);
        per_row.push(json!({"row":i,"position":p.len()-1+i,"cpu_vs_f32_kld":kld(c,r),"cpu_top1_matches_f32":argmax(c)==first,
            "f32_top1":first,"f32_top2":second,"f32_top2_margin":r[first]-r[second]}));
        eprintln!(
            "oxe: f32 reference row {}/{} elapsed {:.3}s",
            i + 1,
            n,
            t.elapsed().as_secs_f64()
        );
        if i + 1 < n {
            reference.step(&s.pool, teacher[i]);
        }
    }
    let reference_s = t.elapsed().as_secs_f64();
    let cb = bytes(&cpu_rows);
    let rb = bytes(&ref_rows);
    write_new(&dir.join("cpu.f32"), &cb);
    write_new(&dir.join("reference.f32"), &rb);
    let manifest = json!({"version":1,"kind":"O62-original-GGUF-full-f32","model_sha256":sha,
        "split_last":split_last,"fixed_teacher":fixed_teacher.is_some(),"prompt_ids":p,"teacher_ids":teacher,"vocab":s.loaded.plan.desc.n_vocab,"n_ctx":n_ctx,"ubatch":ub,
        "threads":s.pool.nthreads(),"effective":model::engine::effective_switches(),"cpu_s":cpu_s,"reference_s":reference_s,
        "cpu_hash":format!("{:016x}",model::weights::hash_bytes(&cb)),"reference_hash":format!("{:016x}",model::weights::hash_bytes(&rb)),"rows":per_row});
    write_new(
        &dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap().as_bytes(),
    );
    println!(
        "{}",
        json!({"event":"f32_export_done","out":dir.display().to_string(),"reference_s":reference_s,"cpu_s":cpu_s})
    );
}

#[cfg(feature = "npu-prefill")]
pub(super) fn compare(a: &Args) {
    if a.get("npu-prefill").is_none() {
        die("explicit --npu-prefill device|emu required");
    }
    let dir = Path::new(
        a.get("ref")
            .unwrap_or_else(|| die("npu-f32-check --ref DIRECTORY required")),
    );
    let m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    if m["version"] != 1 || m["kind"] != "O62-original-GGUF-full-f32" {
        die("unknown f32 reference manifest");
    }
    let p = ids(&m["prompt_ids"]);
    let teacher = ids(&m["teacher_ids"]);
    let vocab = m["vocab"].as_u64().unwrap() as usize;
    if !(33..=512).contains(&p.len()) || !(1..=256).contains(&teacher.len()) || vocab > 1_000_000 {
        die("unbounded f32 manifest");
    }
    let n_ctx = m["n_ctx"].as_u64().unwrap() as usize;
    let ub = m["ubatch"].as_u64().unwrap() as usize;
    if p.len() + teacher.len() >= n_ctx || n_ctx > 4096 || ub == 0 || ub > 1024 {
        die("invalid f32 reference geometry");
    }
    let count = teacher.len().checked_mul(vocab).unwrap();
    let cpu = read_rows(dir, "cpu.f32", count, m["cpu_hash"].as_str().unwrap());
    let reference = read_rows(
        dir,
        "reference.f32",
        count,
        m["reference_hash"].as_str().unwrap(),
    );
    let s = setup(a, n_ctx);
    if s.loaded.ws.info.model_sha256 != m["model_sha256"].as_str().unwrap()
        || vocab != s.loaded.plan.desc.n_vocab
        || m["threads"].as_u64() != Some(s.pool.nthreads() as u64)
        || m["effective"] != model::engine::effective_switches()
        || p.iter().chain(&teacher).any(|&t| t as usize >= vocab)
    {
        die("f32 proof source/config/history mismatch");
    }
    let mut candidate = Engine::new(
        &s.loaded.plan,
        &s.loaded.ws,
        EngineConfig {
            n_ctx,
            logits: true,
            trace: false,
            profile: false,
        },
    )
    .unwrap_or_else(|e| die(&e));
    maybe_npu_prefill(a, &mut candidate);
    let t = Instant::now();
    if m["split_last"] == true {
        candidate.prefill(&s.pool, &p[..p.len() - 1], ub).unwrap();
        candidate.step(&s.pool, *p.last().unwrap());
    } else {
        candidate.prefill(&s.pool, &p, ub).unwrap();
    }
    let npu_prefill_s = t.elapsed().as_secs_f64();
    let calls = candidate.npu_prefill_stats().unwrap().calls;
    let mut rows = Vec::new();
    let mut mean_cpu = 0.;
    let mut mean_npu = 0.;
    for (i, &tok) in teacher.iter().enumerate() {
        let c = &cpu[i * vocab..(i + 1) * vocab];
        let r = &reference[i * vocab..(i + 1) * vocab];
        let x = candidate.logits();
        if x.len() != vocab || x.iter().any(|x| !x.is_finite()) {
            die("nonfinite/mismatched NPU logits");
        }
        let ck = kld(c, r);
        let nk = kld(x, r);
        mean_cpu += ck;
        mean_npu += nk;
        rows.push(json!({"row":i,"position":p.len()-1+i,"cpu_vs_f32_kld":ck,"npu_vs_f32_kld":nk,
            "npu_top1_matches_cpu":argmax(x)==argmax(c),"npu_top1_matches_f32":argmax(x)==argmax(r),"cpu_top1_matches_f32":argmax(c)==argmax(r),
            "cpu_top1":argmax(c),"npu_top1":argmax(x),"f32_top1":argmax(r)}));
        if i + 1 < teacher.len() {
            candidate.step(&s.pool, tok);
        }
    }
    mean_cpu /= teacher.len() as f64;
    mean_npu /= teacher.len() as f64;
    let st = candidate.npu_prefill_stats().unwrap();
    let no_decode_npu = st.calls == calls;
    let mut before = Fnv(0xcbf29ce484222325);
    candidate.write_state(&mut before).unwrap();
    let stats = npu_stats_json(st.clone());
    candidate.release_npu_prefill();
    let mut after = Fnv(0xcbf29ce484222325);
    candidate.write_state(&mut after).unwrap();
    let state_ok = before.0 == after.0;
    let passed = mean_npu <= 1.1 * mean_cpu
        && no_decode_npu
        && state_ok
        && calls > 0
        && st.first_error.is_none();
    let result = json!({"passed":passed,"kind":"O62-NPU-and-CPU-vs-full-f32","mean_cpu_vs_f32_kld":mean_cpu,"mean_npu_vs_f32_kld":mean_npu,
        "gate_multiplier":1.1,"ratio":mean_npu/mean_cpu,"rows":rows,"model_sha256":m["model_sha256"],"prompt_tokens":p.len(),"teacher_ids_hash":sha_ids(&teacher),
        "npu":stats,"decode_npu_calls_zero":no_decode_npu,"release_state_identical":state_ok,"npu_prefill_s_informational":npu_prefill_s,
        "standard_greedy_cases":"separate required gate; not established by teacher forcing"});
    let text = serde_json::to_string_pretty(&result).unwrap();
    if let Some(p) = a.get("out") {
        write_new(Path::new(p), text.as_bytes());
    }
    println!("{text}");
    if !passed {
        std::process::exit(1);
    }
}

/// Standard fixed cases, independent free-running greedy trajectories. A
/// teacher-forced top-1 check is not substituted for sequence identity.
#[cfg(feature = "npu-prefill")]
pub(super) fn greedy(a: &Args) {
    if a.get("npu-prefill").is_none() {
        die("explicit --npu-prefill device|emu required");
    }
    let refs: Vec<_> = a
        .get("refs")
        .unwrap_or_else(|| die("--refs CASE1,CASE2,CASE3 required"))
        .split(',')
        .collect();
    if refs.len() != 3 {
        die("use the three standard cases");
    }
    let n = a.num("n", 128);
    let n_ctx = a.num("ctx", 2048);
    let ub = a.num("ubatch", 256);
    if n == 0 || n > 256 || n_ctx > 4096 || ub == 0 || ub > 1024 {
        die("unbounded greedy proof");
    }
    let f32_control: Option<serde_json::Value> = a
        .get("f32-greedy")
        .map(|path| serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap());
    let s = setup(a, n_ctx);
    let cfg = EngineConfig {
        n_ctx,
        logits: true,
        trace: false,
        profile: false,
    };
    let cpu = Engine::new(&s.loaded.plan, &s.loaded.ws, cfg.clone()).unwrap_or_else(|e| die(&e));
    if let Some(m) = &f32_control {
        if m["version"] != 1
            || m["kind"] != "O63-f32-greedy-split-last"
            || m["model_sha256"] != s.loaded.ws.info.model_sha256
            || m["n_ctx"] != n_ctx
            || m["ubatch"] != ub
            || m["generated_tokens"] != n
            || m["threads"] != s.pool.nthreads()
            || m["effective"] != model::engine::effective_switches()
        {
            die("f32 greedy control identity/config mismatch");
        }
    }
    let mut candidate = Engine::new(&s.loaded.plan, &s.loaded.ws, cfg).unwrap_or_else(|e| die(&e));
    maybe_npu_prefill(a, &mut candidate);
    let tok = tokenizer::Tokenizer::from_gguf(a.model()).unwrap_or_else(|e| die(&e.to_string()));
    let mut cases = Vec::new();
    for path in refs {
        let m: serde_json::Value =
            serde_json::from_slice(&std::fs::read(Path::new(path).join("manifest.json")).unwrap())
                .unwrap();
        let p = ids(&m["tokens"]);
        if p.len() < 33
            || p.len() > 1024
            || p.len() + n >= n_ctx
            || p.iter().any(|&t| t as usize >= s.loaded.plan.desc.n_vocab)
        {
            die("invalid standard greedy fixture");
        }
        cpu.reset();
        candidate.reset();
        // The qualified server's split-last schedule: batch all but the last
        // prompt token, then use decode for the final prompt token.
        cpu.prefill(&s.pool, &p[..p.len() - 1], ub).unwrap();
        candidate.prefill(&s.pool, &p[..p.len() - 1], ub).unwrap();
        let mut cn = cpu.step(&s.pool, *p.last().unwrap());
        let mut nn = candidate.step(&s.pool, *p.last().unwrap());
        let calls = candidate.npu_prefill_stats().unwrap().calls;
        let mut ci = Vec::new();
        let mut ni = Vec::new();
        let mut first = None;
        for i in 0..n {
            ci.push(cn);
            ni.push(nn);
            if cn != nn && first.is_none() {
                first = Some(
                    json!({"row":i,"cpu":cn,"npu":nn,"kld_cpu_vs_npu":kld(cpu.logits(),candidate.logits())}),
                );
            }
            if i + 1 < n {
                cn = cpu.step(&s.pool, cn);
                nn = candidate.step(&s.pool, nn);
            }
        }
        let st = candidate.npu_prefill_stats().unwrap();
        let f32_gate = if let Some(m) = &f32_control {
            let name = Path::new(path).file_name().unwrap().to_string_lossy();
            let c = m["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["case"] == name.as_ref())
                .unwrap_or_else(|| die("missing f32 greedy case"));
            if ids(&c["prompt_ids"]) != p || ids(&c["cpu_ids"]) != ci {
                die("f32 greedy CPU/history changed");
            }
            let f = ids(&c["f32_ids"]);
            if f.len() != n {
                die("f32 greedy length mismatch");
            }
            let cd = ci.iter().zip(&f).position(|(a, b)| a != b).unwrap_or(n);
            let nd = ni.iter().zip(&f).position(|(a, b)| a != b).unwrap_or(n);
            json!({"cpu_first_divergence_from_f32":cd,"npu_first_divergence_from_f32":nd,"npu_at_least_cpu":nd>=cd})
        } else {
            serde_json::Value::Null
        };
        cases.push(json!({"case":Path::new(path).file_name().unwrap().to_string_lossy(),"prompt_tokens":p.len(),"generated_tokens":n,
            "cpu_text":tok.decode(&ci,true),"npu_text":tok.decode(&ni,true),"f32_gate":f32_gate,"cpu_ids":ci,"npu_ids":ni,"identical":ci==ni,"cpu_ids_hash":sha_ids(&ci),"npu_ids_hash":sha_ids(&ni),"first_mismatch":first,"decode_npu_calls_zero":st.calls==calls}));
        eprintln!("oxe: NPU greedy case {}", cases.last().unwrap());
    }
    let st = candidate.npu_prefill_stats().unwrap();
    let passed = cases.iter().all(|c| {
        (if f32_control.is_some() {
            c["f32_gate"]["npu_at_least_cpu"] == true
        } else {
            c["identical"] == true
        }) && c["decode_npu_calls_zero"] == true
    }) && st.calls > 0
        && st.first_error.is_none();
    let stats = npu_stats_json(st);
    candidate.release_npu_prefill();
    let result = json!({"passed":passed,"kind":"W2-three-standard-independent-greedy-cases","cases":cases,"npu":stats,
        "model_sha256":s.loaded.ws.info.model_sha256,"effective":model::engine::effective_switches()});
    let text = serde_json::to_string_pretty(&result).unwrap();
    if let Some(path) = a.get("out") {
        write_new(Path::new(path), text.as_bytes());
    }
    println!("{text}");
    if !passed {
        std::process::exit(1);
    }
}

/// O63: independently generate f32 greedy sequences on the exact standard
/// split-last prompts. Also record a fresh CPU sequence; no accelerator opens.
pub(super) fn greedy_export(a: &Args) {
    if a.get("npu-prefill").is_some() {
        die("f32 greedy export is CPU-only");
    }
    let refs: Vec<_> = a
        .get("refs")
        .unwrap_or_else(|| die("--refs required"))
        .split(',')
        .collect();
    let n = a.num("n", 128);
    let n_ctx = a.num("ctx", 2048);
    let ub = a.num("ubatch", 256);
    if refs.len() != 3 || n == 0 || n > 256 || n_ctx > 4096 || ub == 0 || ub > 1024 {
        die("unbounded f32 greedy export");
    }
    let s = setup(a, n_ctx);
    let (g, sha) = verified_source(a, &s.loaded.ws.info.model_sha256);
    let cfg = EngineConfig {
        n_ctx,
        logits: true,
        trace: false,
        profile: false,
    };
    let cpu = Engine::new(&s.loaded.plan, &s.loaded.ws, cfg.clone()).unwrap_or_else(|e| die(&e));
    let mut reference = Engine::new(&s.loaded.plan, &s.loaded.ws, cfg).unwrap_or_else(|e| die(&e));
    reference
        .enable_f32_reference(g, &sha)
        .unwrap_or_else(|e| die(&e));
    let mut cases = Vec::new();
    for path in refs {
        let m: serde_json::Value =
            serde_json::from_slice(&std::fs::read(Path::new(path).join("manifest.json")).unwrap())
                .unwrap();
        let p = ids(&m["tokens"]);
        if p.len() < 33
            || p.len() > 1024
            || p.len() + n >= n_ctx
            || p.iter().any(|&t| t as usize >= s.loaded.plan.desc.n_vocab)
        {
            die("invalid f32 greedy fixture");
        }
        cpu.reset();
        reference.reset();
        let t = Instant::now();
        let mut seq = Vec::new();
        let mut ci = Vec::new();
        cpu.prefill(&s.pool, &p[..p.len() - 1], ub).unwrap();
        let mut cn = cpu.step(&s.pool, *p.last().unwrap());
        for i in 0..n {
            ci.push(cn);
            if i + 1 < n {
                cn = cpu.step(&s.pool, cn);
            }
        }
        reference.prefill(&s.pool, &p[..p.len() - 1], ub).unwrap();
        let mut rn = reference.step(&s.pool, *p.last().unwrap());
        for i in 0..n {
            seq.push(rn);
            if i + 1 < n {
                rn = reference.step(&s.pool, rn);
            }
            if (i + 1) % 16 == 0 {
                eprintln!(
                    "oxe: f32 greedy {} {}/{} elapsed {:.3}s",
                    path,
                    i + 1,
                    n,
                    t.elapsed().as_secs_f64()
                );
            }
        }
        let first = ci.iter().zip(&seq).position(|(a, b)| a != b).unwrap_or(n);
        cases.push(json!({"case":Path::new(path).file_name().unwrap().to_string_lossy(),"prompt_ids":p,"cpu_ids":ci,"f32_ids":seq,
            "cpu_ids_hash":sha_ids(&ci),"f32_ids_hash":sha_ids(&seq),"cpu_first_divergence":first,"seconds":t.elapsed().as_secs_f64()}));
    }
    let result = json!({"version":1,"kind":"O63-f32-greedy-split-last","model_sha256":sha,"threads":s.pool.nthreads(),
        "n_ctx":n_ctx,"ubatch":ub,"generated_tokens":n,"effective":model::engine::effective_switches(),"cases":cases});
    let path = Path::new(
        a.get("out")
            .unwrap_or_else(|| die("--out NEW_FILE required")),
    );
    write_new(
        path,
        serde_json::to_string_pretty(&result).unwrap().as_bytes(),
    );
    println!(
        "{}",
        json!({"event":"f32_greedy_done","out":path.display().to_string()})
    );
}

/// Bounded diagnostic, not a speed arm: keep each prefix and final-token named
/// tensor so local analysis can distinguish propagation from local roundoff.
pub(super) fn tensor_map(a: &Args) {
    let mode = a
        .get("reference-mode")
        .unwrap_or_else(|| die("--reference-mode cpu|f32|npu required"));
    if !["cpu", "f32", "npu"].contains(&mode) {
        die("invalid reference mode");
    }
    if (mode == "npu") != a.get("npu-prefill").is_some() {
        die("NPU option only allowed for npu tensor map");
    }
    let p = prompt(a);
    let n_ctx = a.num("ctx", 2048);
    let ub = a.num("ubatch", 256);
    if !(33..=64).contains(&p.len()) || n_ctx > 4096 || p.len() >= n_ctx || ub == 0 || ub > 1024 {
        die("bounded tensor map needs33..64 prompt tokens");
    }
    let out = Path::new(
        a.get("out")
            .unwrap_or_else(|| die("--out NEW_DIRECTORY required")),
    );
    std::fs::create_dir(out).unwrap();
    std::fs::set_permissions(out, std::fs::Permissions::from_mode(0o700)).unwrap();
    let s = setup(a, n_ctx);
    if p.iter().any(|&t| t as usize >= s.loaded.plan.desc.n_vocab) {
        die("tensor map token outside vocabulary");
    }
    let mut e = Engine::new(
        &s.loaded.plan,
        &s.loaded.ws,
        EngineConfig {
            n_ctx,
            logits: true,
            trace: true,
            profile: false,
        },
    )
    .unwrap_or_else(|e| die(&e));
    if mode == "f32" {
        let (g, sha) = verified_source(a, &s.loaded.ws.info.model_sha256);
        e.enable_f32_reference(g, &sha).unwrap_or_else(|e| die(&e));
    }
    if mode == "npu" {
        maybe_npu_prefill(a, &mut e);
    }
    let t = Instant::now();
    e.prefill(&s.pool, &p[..p.len() - 1], ub).unwrap();
    let mut records = Vec::new();
    for phase in ["prefix", "last"] {
        if phase == "last" {
            e.step(&s.pool, *p.last().unwrap());
        }
        let dir = out.join(phase);
        std::fs::create_dir(&dir).unwrap();
        let trace = e.trace.as_ref().unwrap().take();
        let total: usize = trace.values().map(|v| v.len() * 4).sum();
        if total > 2usize << 30 {
            die("unbounded tensor trace");
        }
        for (name, row) in trace {
            if !name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-./".contains(&c))
                || name.contains('/')
                || name == ".."
            {
                die("unsafe tensor name");
            }
            let path = dir.join(format!("{name}.f32"));
            write_new(&path, &bytes(&row));
            records.push(json!({"phase":phase,"name":name,"len":row.len(),"nonfinite":row.iter().filter(|x|!x.is_finite()).count(),"file":path.strip_prefix(out).unwrap().display().to_string()}));
        }
        write_new(&dir.join("logits.f32"), &bytes(e.logits()));
    }
    #[cfg(feature = "npu-prefill")]
    let stats = e.npu_prefill_stats().map(npu_stats_json);
    #[cfg(not(feature = "npu-prefill"))]
    let stats: Option<serde_json::Value> = None;
    #[cfg(feature = "npu-prefill")]
    e.release_npu_prefill();
    let result = json!({"kind":"W2-O63-per-tensor-diagnostic","mode":mode,"prompt_ids":p,"n_ctx":n_ctx,"ubatch":ub,
        "model_sha256":s.loaded.ws.info.model_sha256,"effective":model::engine::effective_switches(),"records":records,"npu":stats,"seconds":t.elapsed().as_secs_f64()});
    write_new(
        &out.join("manifest.json"),
        serde_json::to_string_pretty(&result).unwrap().as_bytes(),
    );
    println!(
        "{}",
        json!({"event":"tensor_map_done","mode":mode,"out":out.display().to_string()})
    );
}

fn top2(row: &[f32]) -> (usize, usize) {
    assert!(row.len() >= 2);
    let (mut first, mut second) = if row[0] >= row[1] { (0, 1) } else { (1, 0) };
    for i in 2..row.len() {
        if row[i] > row[first] {
            second = first;
            first = i;
        } else if row[i] > row[second] {
            second = i;
        }
    }
    (first, second)
}
/// Tokenizer-only preparation of declared synthetic corpus text. No model arena,
/// threadpool, device handle or source payload is opened.
pub(super) fn fixture_tokens(a: &Args) {
    let path = a
        .get("files-from")
        .unwrap_or_else(|| die("--files-from synthetic spec JSON required"));
    let raw = std::fs::read(path).unwrap();
    if raw.len() > 1 << 20 {
        die("bounded fixture source required");
    }
    let cases: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let tk = tokenizer::Tokenizer::from_gguf(a.model()).unwrap_or_else(|e| die(&e.to_string()));
    let prefix = a.num("prompt-len", 256);
    let n = a.num("n", 256);
    if prefix < 33 || prefix > 512 || n == 0 || n > 256 {
        die("bounded fixture sizes required");
    }
    let mut result = Vec::new();
    for c in cases.as_array().unwrap() {
        let name = c["case"].as_str().unwrap();
        let text = c["text"].as_str().unwrap();
        let all = tk.encode(text, false, true);
        if all.len() < prefix + n {
            die(&format!(
                "fixture {name} has{} tokens, needs{}",
                all.len(),
                prefix + n
            ));
        }
        result.push(json!({"case":name,"prompt_ids":&all[..prefix],"teacher_ids":&all[prefix..prefix+n],"source_text_hash":format!("{:016x}",model::weights::hash_bytes(text.as_bytes())),"source_tokens":all.len()}));
    }
    write_new(
        Path::new(
            a.get("out")
                .unwrap_or_else(|| die("--out NEW_JSON required")),
        ),
        serde_json::to_string_pretty(&result).unwrap().as_bytes(),
    );
}

/// Decode only the declared local synthetic fixture outputs for human sanity.
pub(super) fn inspect_text(a: &Args) {
    let raw = std::fs::read(
        a.get("files-from")
            .unwrap_or_else(|| die("--files-from result JSON required")),
    )
    .unwrap();
    if raw.len() > 1 << 20 {
        die("bounded result required");
    }
    let m: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let tk = tokenizer::Tokenizer::from_gguf(a.model()).unwrap_or_else(|e| die(&e.to_string()));
    let cases = m["cases"].as_array().unwrap();
    let mut out = Vec::new();
    for c in cases {
        let mut row = serde_json::Map::new();
        row.insert("case".into(), c["case"].clone());
        for key in ["cpu_ids", "npu_ids", "f32_ids"] {
            if c[key].is_array() {
                row.insert(
                    key.replace("_ids", "_text"),
                    json!(tk.decode(&ids(&c[key]), true)),
                );
            }
        }
        out.push(serde_json::Value::Object(row));
    }
    let text = serde_json::to_string_pretty(&out).unwrap();
    if let Some(path) = a.get("out") {
        write_new(Path::new(path), text.as_bytes());
    }
    println!("{text}");
}
