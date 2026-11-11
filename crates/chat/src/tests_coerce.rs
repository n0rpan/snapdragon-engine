//! F105: lenient typed coercion of non-string tool parameters (`parser::coerce_value`).
//! A child module of `tests` (uses its stream / fuzz helpers); kept in its own file so the
//! same patch applies to the MoE and the Flash lineages.

use super::*;

const GP: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

fn typed_tool() -> Tool {
    Tool {
        name: "t".into(),
        description: String::new(),
        parameters: json!({"type":"object","properties":{
            "s":{"type":"string"},
            "str":{"type":"string"},
            "i":{"type":"integer","minimum":0},
            "n":{"type":"number"},
            "b":{"type":"boolean"},
            "nb":{"type":["boolean","null"]},
            "ni":{"anyOf":[{"type":"integer"},{"type":"null"}]},
            "a":{"type":"array","items":{"type":"string"}},
            "o":{"type":"object","properties":{"k":{"type":"integer"}}},
            "e":{"type":"string","enum":["text","markdown","html"]},
            "ei":{"enum":[1,2,3]},
            "ms":{"type":["string","integer"]}
        },"required":["s"]}),
    }
}

fn typed_text(param: &str, raw: &str) -> String {
    format!("Calling.\n\n<tool_call>\n<function=t>\n<parameter=s>\nx\n</parameter>\n<parameter={param}>\n{raw}\n</parameter>\n</function>\n</tool_call>")
}

/// Is `raw` a value the strict grammar (llama.cpp parity) accepts for these types?
fn strict_ok(param: &str, raw: &str) -> bool {
    let strict = cfg_with(vec![typed_tool()], false);
    parser::parse(&strict, GP, &typed_text(param, raw), false).unwrap().tool_calls.len() == 1
}

/// Feed `output` char by char; at every step the streamed calls must equal the final message's
/// first calls exactly (a call never appears and then disappears or changes). Returns the
/// final message.
fn stream_never_shrinks(cfg: &ParserConfig, gp: &str, output: &str) -> ChatMsg {
    let mut s = Session::new(cfg.clone(), gp.to_string(), Box::new(|| "id".into()));
    let mut steps = Vec::new();
    // client view of the calls so far: (name, arguments) by index
    let mut seen: Vec<(String, String)> = Vec::new();
    let mut buf = [0u8; 4];
    for c in output.chars() {
        let d = s.feed(c.encode_utf8(&mut buf), false).unwrap_or_else(|e| panic!("stream error {e:?} on {output:?}"));
        for d in &d {
            if let Some(i) = d.tool_call_index {
                // only complete calls stream, each in one piece: a delta on an earlier call
                // would mean it changed
                assert_eq!(i, seen.len(), "a streamed call changed at index {i} on {output:?}");
                seen.push((d.tool_call_delta.name.clone(), d.tool_call_delta.arguments.clone()));
            }
        }
        steps.push(Json::Array(d.iter().map(|d| d.to_delta_json()).collect()));
    }
    let d = s.feed("", true).unwrap_or_else(|e| panic!("final error {e:?} on {output:?}"));
    steps.push(Json::Array(d.iter().map(|d| d.to_delta_json()).collect()));
    let fin = s.msg().clone();
    assert_eq!(assemble(&steps), msg_view(&fin), "stream != final on {output:?}");
    let fc: Vec<(String, String)> = fin.tool_calls.iter().map(|c| (c.name.clone(), c.arguments.clone())).collect();
    assert!(seen.len() <= fc.len() && fc[..seen.len()] == seen[..], "final calls {fc:?} drop or change streamed {seen:?}");
    fin
}

/// One case: lenient final parse, char-by-char stream, truncation fuzz, incremental == full.
/// `want` None: the call fails (as before F105) and its text comes back as content.
fn check(param: &str, raw: &str, want: Option<Json>) {
    let cfg = cfg_with(vec![typed_tool()], true);
    let text = typed_text(param, raw);
    let m = parser::parse(&cfg, GP, &text, false).unwrap();
    match &want {
        Some(w) => {
            assert_eq!(m.tool_calls.len(), 1, "{param}={raw:?}: {m:?}");
            let args: Json = serde_json::from_str(&m.tool_calls[0].arguments).unwrap_or_else(|e| panic!("{param}={raw:?}: bad JSON {e}: {}", m.tool_calls[0].arguments));
            assert_eq!(&args[param], w, "{param}={raw:?}");
            assert_eq!(args["s"], "x");
            assert_eq!(m.content, "Calling.");
        }
        None => {
            assert!(m.tool_calls.is_empty(), "{param}={raw:?} must stay a failure: {m:?}");
            assert!(m.content.contains("<tool_call>") && m.content.contains(raw.trim()), "{param}={raw:?}: text fallback {m:?}");
        }
    }
    let fin = stream_never_shrinks(&cfg, GP, &text);
    assert_eq!(fin.tool_calls.len(), m.tool_calls.len(), "{param}={raw:?}");
    if let (Some(a), Some(b)) = (fin.tool_calls.first(), m.tool_calls.first()) {
        assert_eq!(a.arguments, b.arguments);
    }
    fuzz_truncations(&cfg, GP, &text);
    for ch in [vec![1], vec![3], vec![7]] {
        diff_sessions(&cfg, GP, &text, &ch);
    }
    // strict mode (llama.cpp parity) is unchanged: it accepts exactly what the grammar accepts,
    // verbatim
    // (lenient accepts a superset; it differs from strict only where it normalizes `1.0` -> `1`)
    let strict = cfg_with(vec![typed_tool()], false);
    let sm = parser::parse(&strict, GP, &text, false).unwrap();
    if strict_ok(param, raw) {
        let sv: Json = serde_json::from_str(&sm.tool_calls[0].arguments).unwrap();
        let w = want.as_ref().unwrap_or_else(|| panic!("{param}={raw:?}: strict accepts what lenient rejects"));
        assert!(&sv[param] == w || (sv[param].as_f64().is_some() && sv[param].as_f64() == w.as_f64()), "strict {param}={raw:?}: {sm:?}");
        if !sm.tool_calls[0].arguments.contains(&format!("\"{param}\":\"")) {
            assert!(sm.tool_calls[0].arguments.contains(&format!("\"{param}\":{raw}")), "strict is verbatim: {param}={raw:?}: {sm:?}");
        }
    } else {
        assert!(sm.tool_calls.is_empty(), "strict {param}={raw:?}: {sm:?}");
    }
}

#[test]
fn coerce_integer() {
    for (raw, want) in [
        ("5", Some(json!(5))),
        ("-3", Some(json!(-3))),
        ("\"5\"", Some(json!(5))),
        ("\"-12\"", Some(json!(-12))),
        ("1.0", Some(json!(1))),
        ("\"1.0\"", Some(json!(1))),
        ("2e1", Some(json!(20))),
        ("\"2.000\"", Some(json!(2))),
        ("-0.0", Some(json!(0))),
        ("9007199254740992.0", Some(json!(9007199254740992i64))),
        ("123456789012345678901234567890", Some(json!(123456789012345678901234567890.0f64))), // strict-valid, verbatim
        // the strict grammar accepts any JSON number for an integer (as llama.cpp does): a
        // non-integral one is passed on verbatim, as before (`1e400` too, not tested: not JSON-
        // representable as f64 by serde)
        ("1.5", Some(json!(1.5))),
        // ambiguous: a failure, as before
        ("\"1.5\"", None),
        ("\"9007199254740994.0\"", None),
        ("0080", None),
        ("\"0080\"", None),
        ("0x10", None),
        ("+5", None),
        ("five", None),
        ("\"\"", None),
        ("\"\\\"5\\\"\"", None),
        ("NaN", None),
        ("True", None),
        ("None", None),
        ("5 apples", None),
        ("[5]", None),
    ] {
        check("i", raw, want);
    }
}

#[test]
fn coerce_number() {
    for (raw, want) in [
        ("2.5", Some(json!(2.5))),
        ("\"2.5\"", Some(json!(2.5))),
        ("\"30\"", Some(json!(30))),
        ("1.0", Some(json!(1.0))), // a number parameter keeps its form
        ("\"-1e-3\"", Some(json!(-0.001))),
        ("\"Infinity\"", None),
        ("\"1,5\"", None),
        ("", None),
    ] {
        check("n", raw, want);
    }
}

#[test]
fn coerce_bool_and_null() {
    for (p, raw, want) in [
        ("b", "True", Some(json!(true))),
        ("b", "False", Some(json!(false))),
        ("b", "TRUE", Some(json!(true))),
        ("b", "false", Some(json!(false))),
        ("b", "\"false\"", Some(json!(false))),
        ("b", "\"True\"", Some(json!(true))),
        ("b", "None", None),
        ("b", "null", None),
        ("b", "yes", None),
        ("b", "1", None),
        ("b", "0", None),
        ("b", "Falsey", None),
        ("b", "False False", None),
        ("nb", "None", Some(json!(null))),
        ("nb", "\"None\"", Some(json!(null))),
        ("nb", "null", Some(json!(null))),
        ("nb", "\"null\"", Some(json!(null))),
        ("nb", "False", Some(json!(false))),
        ("nb", "none", None),
        ("nb", "NULL", None),
        ("ni", "None", Some(json!(null))),
        ("ni", "\"7\"", Some(json!(7))),
        ("ni", "True", None),
    ] {
        check(p, raw, want);
    }
}

#[test]
fn coerce_nested_json_from_string() {
    for (p, raw, want) in [
        ("a", "[\"x\", \"y\"]", Some(json!(["x", "y"]))),
        ("a", "\"[\\\"x\\\", \\\"y\\\"]\"", Some(json!(["x", "y"]))),
        ("a", "\"[]\"", Some(json!([]))),
        ("a", "  [\"x\"]  ", Some(json!(["x"]))),
        ("o", "{\"k\": 1}", Some(json!({"k": 1}))),
        ("o", "\"{\\\"k\\\": 1}\"", Some(json!({"k": 1}))),
        ("o", "\"{\\\"k\\\": {\\\"deep\\\": [1, true, null]}}\"", Some(json!({"k": {"deep": [1, true, null]}}))),
        // Python containers, wrong container type, broken JSON, double quoting: failures
        ("a", "['x', 'y']", None),
        ("a", "\"{\\\"k\\\": 1}\"", None),
        ("o", "\"[1]\"", None),
        ("o", "\"{\\\"k\\\": 1\"", None),
        ("o", "{\"k\": True}", None),
        ("a", "x, y", None),
        ("a", "\"\\\"[1]\\\"\"", None),
    ] {
        check(p, raw, want);
    }
}

#[test]
fn coerce_enum() {
    // a string enum is string-typed: the bare word is already a plain string value (no
    // coercion; a quoted word stays the string with its quotes)
    check("e", "markdown", Some(json!("markdown")));
    check("e", "\"markdown\"", Some(json!("\"markdown\"")));
    check("e", "True", Some(json!("True")));
    // a non-string enum takes the integer rules
    check("ei", "2", Some(json!(2)));
    check("ei", "\"2\"", Some(json!(2)));
    check("ei", "2.0", Some(json!(2))); // an all-integer enum is integer-typed: 2.0 -> 2
    check("ei", "two", None);
}

#[test]
fn string_typed_parameters_are_never_coerced() {
    // string-only: the text verbatim (no trimming, no unquoting)
    for raw in ["\"5\"", "True", "None", "1.0", "\"[1]\"", "  padded  ", "{\"k\": 1}", "null"] {
        check("str", raw, Some(json!(raw)));
    }
    // string-capable mixed type: JSON of the other type when it parses strictly, else the text
    for (raw, want) in [("\"5\"", json!("\"5\"")), ("True", json!("True")), ("1.0", json!(1.0)), ("5", json!(5)), (" 5 ", json!(" 5 ")), ("\"[1]\"", json!("\"[1]\""))] {
        check("ms", raw, Some(want));
    }
}

#[test]
fn coerce_whitespace_trimming() {
    for (p, raw, want) in [
        ("i", " 5", Some(json!(5))),
        ("i", "5 ", Some(json!(5))),
        ("i", "\t5\r", Some(json!(5))),
        ("i", "\n5", Some(json!(5))),
        ("i", "5\n", Some(json!(5))),
        ("i", "\n\n 5 \n", Some(json!(5))),
        ("i", "\" 5 \"", Some(json!(5))),
        ("b", "  False\r", Some(json!(false))),
        ("nb", "\nNone\n", Some(json!(null))),
        ("o", "\n{\"k\": 1}\n", Some(json!({"k": 1}))),
        ("i", "   ", None),
        ("i", "5\n6", None),
    ] {
        check(p, raw, want);
    }
}

/// The value close tag is exact (`\n</parameter>\n`): a coerced value is withheld until it is
/// closed; a stream that ends inside the value or its close tag is a failure, as before.
#[test]
fn coerce_needs_the_close_tag() {
    let cfg = cfg_with(vec![typed_tool()], true);
    for raw in ["False", "\"5\"", " 5", "\"[\\\"x\\\"]\""] {
        let p = if raw.contains('[') { "a" } else if raw == "False" { "b" } else { "i" };
        let full = typed_text(p, raw);
        let vstart = full.find(&format!("<parameter={p}>\n")).unwrap() + p.len() + 13;
        let close = full.find("\n</parameter>\n</function>").unwrap();
        // partial parses inside the value or its close tag carry the call without the value
        for cut in vstart..close + "\n</parameter>".len() {
            if !full.is_char_boundary(cut) {
                continue;
            }
            let (m, _, complete) = parser::parse_full(&cfg, &format!("{GP}{}", &full[..cut]), true).unwrap();
            assert_eq!(complete, 0, "{raw:?} cut {cut}");
            if let Some(c) = m.tool_calls.first() {
                assert!(c.arguments.ends_with(&format!("\"{p}\":")) || c.arguments.ends_with(&format!("\"{p}\":{}", &full[vstart..cut])), "{raw:?} cut {cut}: {c:?}");
            }
            // a final parse there: no call (the value is not closed), its text is kept
            let fm = parser::parse(&cfg, GP, &full[..cut], false).unwrap();
            assert!(fm.tool_calls.is_empty(), "{raw:?} final cut {cut}: {fm:?}");
            assert!(fm.content.contains("<tool_call>"));
        }
        stream_never_shrinks(&cfg, GP, &full);
    }
}

/// The pinned 01:45 G0′ text (`g0prime_text`) with every replaceAll spelling: char-by-char
/// stream never shows a call that later vanishes; strict mode unchanged.
#[test]
fn g0prime_replace_all_spellings_stream() {
    let cfg = cfg_with(vec![opencode_edit_tool()], true);
    let strict = cfg_with(vec![opencode_edit_tool()], false);
    for (b, want) in [("False", Some("false")), ("True", Some("true")), ("false", Some("false")), ("\"false\"", Some("false")), ("FALSE", Some("false")), (" False ", Some("false")), ("Fals", None), ("False.", None), ("0", None)] {
        let text = g0prime_text(b);
        let fin = stream_never_shrinks(&cfg, GP, &text);
        match want {
            Some(w) => {
                assert_eq!(fin.tool_calls.len(), 1, "{b}");
                let args: Json = serde_json::from_str(&fin.tool_calls[0].arguments).unwrap();
                assert_eq!(args["replaceAll"].to_string(), w, "{b}");
                assert_eq!(args["path"], "/work/project/parse_port.py");
            }
            None => {
                assert!(fin.tool_calls.is_empty(), "{b}");
                assert!(fin.content.contains("<parameter=replaceAll>"), "{b}");
            }
        }
        fuzz_truncations(&cfg, GP, &text);
        // strict: replaceAll (optional) before the required ones is out of grammar order anyway
        let sm = parser::parse(&strict, GP, &text, false).unwrap();
        assert!(sm.tool_calls.is_empty(), "strict {b}");
    }
}

// ---------------------------------------------------------------------------------------
// fuzz corpus from the G0′ captures (tests/fixtures/g0prime_corpus.jsonl, made by
// tools/make_g0prime_corpus.py from the fixed-run request captures: OpenCode's tool schemas and
// every assistant turn rendered back to Qwen3-Coder text)

fn g0prime_corpus() -> (Vec<Tool>, Vec<String>) {
    let data = include_str!("../tests/fixtures/g0prime_corpus.jsonl");
    let (mut tools, mut outs) = (Vec::new(), Vec::new());
    for line in data.lines() {
        let v: Json = serde_json::from_str(line).unwrap();
        match v["kind"].as_str() {
            Some("tools") => {
                for t in v["tools"].as_array().unwrap() {
                    tools.push(Tool { name: t["name"].as_str().unwrap().into(), description: String::new(), parameters: t["parameters"].clone() });
                }
            }
            Some("output") => outs.push(v["output"].as_str().unwrap().to_string()),
            _ => panic!("bad corpus line {line}"),
        }
    }
    (tools, outs)
}

/// Spellings per parameter type: (raw, expected JSON or None = the call must fail).
fn spellings(schema: &Json) -> Vec<(&'static str, Option<Json>)> {
    match schema["type"].as_str() {
        Some("boolean") => vec![("True", Some(json!(true))), ("False", Some(json!(false))), ("\"true\"", Some(json!(true))), (" false\r", Some(json!(false))), ("yes", None), ("Tru", None)],
        Some("integer") => vec![("\"5\"", Some(json!(5))), ("5.0", Some(json!(5))), ("\" 12 \"", Some(json!(12))), ("\"5.5\"", None), ("05", None), ("None", None)],
        Some("number") => vec![("\"2.5\"", Some(json!(2.5))), ("30", Some(json!(30))), ("\"x\"", None)],
        Some("array") => vec![("\"[]\"", Some(json!([]))), ("[]", Some(json!([]))), ("['a']", None)],
        _ => vec![],
    }
}

#[test]
fn fuzz_g0prime_corpus_typed_spellings() {
    let (tools, outs) = g0prime_corpus();
    assert_eq!(tools.len(), 12);
    assert!(outs.len() >= 8);
    let lenient = cfg_with(tools.clone(), true);
    let strict = cfg_with(tools.clone(), false);
    let mut originals = outs.clone();
    for b in ["False", "True", "false"] {
        originals.push(g0prime_text(b)); // the captured 01:45 failing edit (and its fix)
    }
    let (mut n_cases, mut n_cuts, mut n_ok, mut n_fail) = (0, 0, 0, 0);
    // 1) the captured turns themselves: same calls in both modes, stream == final
    for out in &outs {
        let a = parser::parse(&lenient, GP, out, false).unwrap();
        let b = parser::parse(&strict, GP, out, false).unwrap();
        assert!(!a.tool_calls.is_empty(), "{out:?}");
        // strict drops a call whose optional parameter comes before a required one (turn 6:
        // replaceAll first, grammar order); otherwise identical
        if !out.contains("<parameter=replaceAll>\nfalse\n</parameter>\n<parameter=path>") {
            assert_eq!(a.tool_calls, b.tool_calls, "{out:?}");
        }
        stream_never_shrinks(&lenient, GP, out);
        n_cuts += fuzz_truncations(&lenient, GP, out);
    }
    // 2) every captured call with each typed parameter of its tool added in every spelling
    for out in &originals {
        let base = parser::parse(&lenient, GP, out, false).unwrap();
        let ends: Vec<usize> = out.match_indices("</function>").map(|(i, _)| i).collect();
        assert_eq!(ends.len(), base.tool_calls.len(), "{out:?}");
        for (ci, &end) in ends.iter().enumerate() {
            let tool = tools.iter().find(|t| t.name == base.tool_calls[ci].name).unwrap();
            for (pname, schema) in tool.parameters["properties"].as_object().unwrap() {
                if out[..end].rfind("<function=").map(|f| out[f..end].contains(&format!("<parameter={pname}>"))).unwrap_or(false) {
                    continue; // already written by the model
                }
                for (raw, want) in spellings(schema) {
                    let text = format!("{}<parameter={pname}>\n{raw}\n</parameter>\n{}", &out[..end], &out[end..]);
                    let m = parser::parse(&lenient, GP, &text, false).unwrap();
                    match &want {
                        Some(w) => {
                            assert_eq!(m.tool_calls.len(), base.tool_calls.len(), "{pname}={raw:?} in {text:?}");
                            let args: Json = serde_json::from_str(&m.tool_calls[ci].arguments).unwrap();
                            assert_eq!(&args[pname.as_str()], w, "{pname}={raw:?}");
                            for (k, v) in serde_json::from_str::<Json>(&base.tool_calls[ci].arguments).unwrap().as_object().unwrap() {
                                assert_eq!(&args[k.as_str()], v, "{pname}={raw:?}: other argument {k} changed");
                            }
                            n_ok += 1;
                        }
                        None => {
                            // the call and everything after it come back as text, nothing lost
                            assert_eq!(m.tool_calls.len(), ci, "{pname}={raw:?}");
                            assert!(m.content.contains(&format!("<parameter={pname}>\n{raw}")), "{pname}={raw:?}");
                            n_fail += 1;
                        }
                    }
                    let fin = stream_never_shrinks(&lenient, GP, &text);
                    assert_eq!(fin.tool_calls, m.tool_calls.iter().map(|c| ToolCall { id: "id".into(), ..c.clone() }).collect::<Vec<_>>(), "{pname}={raw:?}");
                    n_cuts += fuzz_truncations(&lenient, GP, &text);
                    // strict: the call carries the value only if the grammar accepts it verbatim
                    let sm = parser::parse(&strict, GP, &text, false).unwrap();
                    if sm.tool_calls.len() > ci {
                        assert!(sm.tool_calls[ci].arguments.contains(&format!("\"{pname}\":{raw}")), "strict {pname}={raw:?}");
                    }
                    n_cases += 1;
                }
            }
        }
    }
    eprintln!("g0prime corpus: {} turns, {n_cases} typed spellings ({n_ok} coerced, {n_fail} failures kept), {n_cuts} truncations", outs.len());
    assert!(n_ok > 50 && n_fail > 20);
}
