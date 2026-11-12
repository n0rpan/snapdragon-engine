//! Flash production configuration (`msg_split` on, small cancel ubatch) and the request ledger,
//! model-free: the stub engine only.

// LEAD: make these pub(crate) in tests.rs (private today):
//   fns:    model, greedy, scratch_dir, stub_worker, prefills
//   struct: CancelStub (pub(crate)), and its fields inner, flag, at, calls (pub(crate))
// msg_params is deliberately not used: the flash tests spell out GenParams with struct update.

use crate::engine::StubEngine;
use crate::generate::{GenError, GenParams, Worker, CANCEL_UBATCHES};
use crate::tests::{greedy, model, prefills, scratch_dir, stub_worker, CancelStub};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Cancel-check ubatch of these tests (the prefill is cut every `U * CANCEL_UBATCHES` tokens).
const U: usize = 16;

/// The Flash message layout: messages at 0, 400 (first user), 900 (last user, the query); the
/// prompt has 1000 tokens, so its last token is at 999 (the single-token decode-step call).
fn flash_params() -> GenParams {
    GenParams { first_user: Some(400), query_start: true, msg_starts: vec![0, 400, 900], ..greedy() }
}

/// Cancel mid-prefill under msg_split with a small cancel ubatch, then a retry on the same worker.
/// The first segment is [0,400); its prefill is cut at multiples of U*CANCEL_UBATCHES = 32 from
/// 0, so the flag raised by the 3rd prefill call (which runs its chunk, 64..96) stops the
/// request at 3*32 = 96. The retry resumes from there on the same cut grid, so its prefill calls
/// from 96 on are exactly the uncancelled run's, and its reply is identical.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn flash_msg_split_cancel_resumes_on_schedule() {
    let Some(m) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let eos = tok.eos.unwrap();
    let script = tok.encode("a reply of a few tokens", false, true);
    let prompt: Vec<u32> = (0..1000u32).map(|i| 100 + i % 5000).collect();
    let gp = flash_params();
    // reference: no cancel
    let mut reference = Worker::new(StubEngine::new(4096, tok.n_vocab(), eos, script.clone()), 4);
    reference.ubatch = U;
    reference.msg_split = true;
    let ref_tokens = reference.generate(&tok, &prompt, &gp, &mut |_| true).unwrap().tokens;
    let ref_calls = prefills(&reference.engine.calls, 0);
    // cancelled run
    let flag = Arc::new(AtomicBool::new(false));
    let eng = CancelStub { inner: StubEngine::new(4096, tok.n_vocab(), eos, script.clone()), flag: flag.clone(), at: 3, calls: 0 };
    let mut w = Worker::new(eng, 4);
    w.ubatch = U;
    w.msg_split = true;
    let mut cgp = gp.clone();
    cgp.cancel = Some(flag.clone());
    assert!(matches!(w.generate(&tok, &prompt, &cgp, &mut |_| true), Err(GenError::Cancelled)));
    let stop = 3 * U * CANCEL_UBATCHES;
    assert_eq!(w.engine.inner.tokens, w.cache.tokens, "engine and cache agree after a cancel");
    assert_eq!(w.cache.tokens.len(), stop, "stopped at the 3rd cut");
    assert_eq!(w.cache.tokens[..], prompt[..stop]);
    // retry: the live partial prefill is a prefix of the prompt, so it is resumed whole
    flag.store(false, Ordering::Relaxed);
    w.engine.at = usize::MAX;
    let mark = w.engine.inner.calls.len();
    let r = w.generate(&tok, &prompt, &cgp, &mut |_| true).unwrap();
    assert_eq!(r.timings.cache_n, stop);
    assert_eq!(r.tokens, ref_tokens);
    // the retry's calls are the reference's calls that start at or after the stop position
    let want: Vec<(usize, usize)> = ref_calls.iter().copied().filter(|c| c.0 >= stop).collect();
    assert_eq!(prefills(&w.engine.inner.calls, mark), want);
}

/// Without a cancel: every prefill call of a msg_split request starts on its segment's own cut
/// grid (segment start + k*U*CANCEL_UBATCHES) and ends inside the segment, so no cut moves a
/// ubatch boundary and no call crosses a message start. The single-token call at 999 is the
/// decode step for the last prompt token and is excluded.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn flash_msg_split_cancel_cut_never_moves_a_chunk_boundary() {
    let Some(m) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let eos = tok.eos.unwrap();
    let script = tok.encode("a reply of a few tokens", false, true);
    let prompt: Vec<u32> = (0..1000u32).map(|i| 100 + i % 5000).collect();
    let mut w = Worker::new(StubEngine::new(4096, tok.n_vocab(), eos, script), 4);
    w.ubatch = U;
    w.msg_split = true;
    w.generate(&tok, &prompt, &flash_params(), &mut |_| true).unwrap();
    let calls: Vec<(usize, usize)> = prefills(&w.engine.calls, 0).into_iter().filter(|c| c.0 != 999).collect();
    // every prompt token before the last one is prefilled once
    assert_eq!(calls.iter().map(|c| c.1).sum::<usize>(), 999);
    let step = U * CANCEL_UBATCHES;
    let segs = [(0usize, 400usize), (400, 900), (900, 999)];
    for &(start, n) in &calls {
        let &(s, e) = segs.iter().find(|&&(s, e)| start >= s && start < e).expect("call outside every segment");
        assert!(start + n <= e, "call ({start}, {n}) crosses the message start {e}");
        assert_eq!((start - s) % step, 0, "call at {start} is off the cut grid of the segment at {s}");
    }
}

/// Cancel mid-prefill, another short request, then the retry, as
/// `cancelled_partial_prefill_survives_another_request`, but under msg_split: the partial prefill
/// is parked by the other request and the retry resumes it whole.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn flash_cancel_then_other_request_then_retry_msg_split() {
    let Some(m) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let eos = tok.eos.unwrap();
    let script = tok.encode("a reply of a few tokens", false, true);
    let prompt: Vec<u32> = (0..1000u32).map(|i| 100 + i % 5000).collect();
    let flag = Arc::new(AtomicBool::new(false));
    let eng = CancelStub { inner: StubEngine::new(4096, tok.n_vocab(), eos, script.clone()), flag: flag.clone(), at: 5, calls: 0 };
    let mut w = Worker::new(eng, 8);
    w.ubatch = U;
    w.msg_split = true;
    w.cache.park_budget = 1 << 30;
    w.cache.park_min = 64;
    let mut gp = greedy();
    gp.cancel = Some(flag.clone());
    assert!(matches!(w.generate(&tok, &prompt, &gp, &mut |_| true), Err(GenError::Cancelled)));
    let partial = w.cache.tokens.len();
    assert!(partial >= 64, "partial prefill of {partial} tokens");
    // the user's next, unrelated short request
    flag.store(false, Ordering::Relaxed);
    w.engine.at = usize::MAX;
    let short: Vec<u32> = (0..20u32).map(|i| 7000 + i).collect();
    w.generate(&tok, &short, &gp, &mut |_| true).unwrap();
    assert_eq!(w.cache.parked_lengths(), vec![partial], "the partial session was parked");
    // the retry resumes the partial prefill
    let r = w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
    assert_eq!(r.timings.cache_n, partial);
    assert_eq!(w.cache.stats.unparks, 1);
}

/// Request ledger (`w.ledger`): one JSON line per successful generation, numbers and enum labels
/// only. Line 2 shares a prefix with line 1 (same anchor as `msg_split_resident_second_question`),
/// so its cache_n is positive. No key outside the ledger's documented set, so no prompt text.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn ledger_writes_one_numeric_line_per_generation() {
    const ALLOWED: &[&str] = &[
        "t", "n_prompt", "cache_n", "cache_src", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "stop", "draft_n",
        "draft_n_accepted", "parks", "unparks", "park_ms", "unpark_ms", "park_evictions", "checkpoints", "parked", "parked_mib", "resume_src", "resume_ms",
    ];
    let Some(m) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let dir = scratch_dir("flash-ledger");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ledger.jsonl");
    let mut w = stub_worker(&tok, 8);
    w.ledger = Some(std::fs::OpenOptions::new().create(true).append(true).open(&path).unwrap());
    let p1: Vec<u32> = (100..140).collect();
    w.generate(&tok, &p1, &flash_params_small(), &mut |_| true).unwrap();
    let mut p2 = p1[..19].to_vec();
    p2.extend(300..321);
    let r2 = w.generate(&tok, &p2, &flash_params_small(), &mut |_| true).unwrap();
    assert!(r2.timings.cache_n > 0);
    drop(w);
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<serde_json::Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.len(), 2, "{text}");
    for line in &lines {
        for k in line.as_object().unwrap().keys() {
            assert!(ALLOWED.contains(&k.as_str()), "unexpected ledger key {k}: {line}");
        }
        for k in ["n_prompt", "cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms"] {
            assert!(line[k].is_number(), "{k} is not a number: {line}");
        }
        assert!(line["stop"].is_string(), "{line}");
    }
    assert!(lines[1]["cache_n"].as_u64().unwrap() > 0, "{}", lines[1]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The message layout of `msg_split_resident_second_question` (anchor 17, query 30), for the ledger
/// test: GenParams with the same hints on both requests.
fn flash_params_small() -> GenParams {
    GenParams { first_user: Some(17), query_start: true, msg_starts: vec![0, 17, 30], ..greedy() }
}

/// A cancelled generation returns before the ledger write (the line is written only at the end of
/// `generate`, on success), so a cancel leaves the ledger file empty.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn cancelled_generation_writes_no_ledger_line() {
    let Some(m) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let eos = tok.eos.unwrap();
    let script = tok.encode("a reply of a few tokens", false, true);
    let prompt: Vec<u32> = (0..1000u32).map(|i| 100 + i % 5000).collect();
    let dir = scratch_dir("flash-ledger-cancel");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ledger.jsonl");
    let flag = Arc::new(AtomicBool::new(false));
    let eng = CancelStub { inner: StubEngine::new(4096, tok.n_vocab(), eos, script), flag: flag.clone(), at: 3, calls: 0 };
    let mut w = Worker::new(eng, 4);
    w.ubatch = U;
    w.ledger = Some(std::fs::OpenOptions::new().create(true).append(true).open(&path).unwrap());
    let mut gp = greedy();
    gp.cancel = Some(flag.clone());
    assert!(matches!(w.generate(&tok, &prompt, &gp, &mut |_| true), Err(GenError::Cancelled)));
    drop(w);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "");
    let _ = std::fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------------ engine error mid-request

/// A stub whose `at`-th prefill or decode call ADVANCES the engine and then fails, as the Flash
/// engine does when an expert read of the chunk fails after the graph already moved its position.
struct FailStub {
    inner: StubEngine,
    at: usize,
    calls: usize,
}

impl crate::engine::Engine for FailStub {
    type Snapshot = <StubEngine as crate::engine::Engine>::Snapshot;
    fn n_ctx(&self) -> usize {
        self.inner.n_ctx()
    }
    fn n_vocab(&self) -> usize {
        self.inner.n_vocab()
    }
    fn n_past(&self) -> usize {
        self.inner.n_past()
    }
    fn reset(&mut self) {
        self.inner.reset()
    }
    fn prefill(&mut self, t: &[u32]) -> crate::engine::Result<&[f32]> {
        self.calls += 1;
        if self.calls == self.at {
            self.inner.prefill(t)?;
            return Err(crate::engine::EngineError("flash: expert read failed".into()));
        }
        self.inner.prefill(t)
    }
    fn decode(&mut self, t: u32) -> crate::engine::Result<&[f32]> {
        self.calls += 1;
        if self.calls == self.at {
            self.inner.decode(t)?;
            return Err(crate::engine::EngineError("flash: expert read failed".into()));
        }
        self.inner.decode(t)
    }
    fn snapshot(&mut self) -> crate::engine::Result<Self::Snapshot> {
        self.inner.snapshot()
    }
    fn restore(&mut self, s: &Self::Snapshot) -> crate::engine::Result<()> {
        self.inner.restore(s)
    }
}

/// Review finding on ticket 052: an engine error after the engine advanced (engine n_past 96, cache
/// 64 in the stub case) left the engine ahead of the cache, and a retry of the same prompt was
/// answered as a live continuation on top of the invalid chunk. Now the state is dropped: engine and
/// cache are both empty after the error, the retry recomputes everything and equals an undisturbed run.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn engine_error_mid_prefill_drops_the_state_and_the_retry_starts_clean() {
    let Some(m) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let eos = tok.eos.unwrap();
    let script = tok.encode("a reply of a few tokens", false, true);
    let prompt: Vec<u32> = (0..1000u32).map(|i| 100 + i % 5000).collect();
    for msg_split in [true, false] {
        let gp = if msg_split { flash_params() } else { greedy() };
        let mut reference = Worker::new(StubEngine::new(4096, tok.n_vocab(), eos, script.clone()), 4);
        reference.ubatch = U;
        reference.msg_split = msg_split;
        let ref_tokens = reference.generate(&tok, &prompt, &gp, &mut |_| true).unwrap().tokens;
        // the 3rd prefill call advances the engine to 96 tokens and fails; the cache knows 64
        let eng = FailStub { inner: StubEngine::new(4096, tok.n_vocab(), eos, script.clone()), at: 3, calls: 0 };
        let mut w = Worker::new(eng, 4);
        w.ubatch = U;
        w.msg_split = msg_split;
        assert!(matches!(w.generate(&tok, &prompt, &gp, &mut |_| true), Err(GenError::Engine(_))));
        assert!(w.engine.inner.tokens.is_empty(), "engine reset after the error ({msg_split})");
        assert!(w.cache.tokens.is_empty(), "cache reset after the error ({msg_split})");
        assert_eq!(crate::engine::Engine::n_past(&w.engine), 0);
        // retry: nothing is reused, the reply is the undisturbed one
        w.engine.at = usize::MAX;
        let mark = w.engine.inner.calls.len();
        let r = w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
        assert_eq!(r.timings.cache_n, 0, "no continuation on the failed chunk ({msg_split})");
        assert_eq!(r.tokens, ref_tokens);
        assert_eq!(prefills(&w.engine.inner.calls, mark)[0].0, 0, "the retry prefills from position 0");
    }
}

/// The same for an error in the decode loop: the state is dropped, the next request (the same
/// prompt) is computed from scratch and equals the reference.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn engine_error_mid_decode_drops_the_state() {
    let Some(m) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let eos = tok.eos.unwrap();
    let script = tok.encode("a reply of a few tokens", false, true);
    let prompt: Vec<u32> = (0..200u32).map(|i| 100 + i % 5000).collect();
    let gp = flash_params_short();
    let mut reference = Worker::new(StubEngine::new(4096, tok.n_vocab(), eos, script.clone()), 4);
    reference.ubatch = U;
    reference.msg_split = true;
    let ref_tokens = reference.generate(&tok, &prompt, &gp, &mut |_| true).unwrap().tokens;
    // FailStub counts prefill and decode calls only
    let n_calls = reference.engine.calls.iter().filter(|c| c.0 == "prefill" || c.0 == "decode").count();
    assert_eq!(reference.engine.calls.iter().filter(|c| c.0 == "decode").count() >= 2, true, "at least two decode steps");
    assert!(ref_tokens.len() >= 3, "the script decodes a few tokens");
    // fail on the second-to-last engine call of the request: inside the decode loop
    let eng = FailStub { inner: StubEngine::new(4096, tok.n_vocab(), eos, script.clone()), at: n_calls - 1, calls: 0 };
    let mut w = Worker::new(eng, 4);
    w.ubatch = U;
    w.msg_split = true;
    assert!(matches!(w.generate(&tok, &prompt, &gp, &mut |_| true), Err(GenError::Engine(_))));
    assert!(w.engine.inner.tokens.is_empty() && w.cache.tokens.is_empty());
    w.engine.at = usize::MAX;
    let r = w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
    assert_eq!(r.timings.cache_n, 0);
    assert_eq!(r.tokens, ref_tokens);
}

/// Messages at 0, 50 (first user) and 150 (the query) of a 200-token prompt.
fn flash_params_short() -> GenParams {
    GenParams { first_user: Some(50), query_start: true, msg_starts: vec![0, 50, 150], ..greedy() }
}
