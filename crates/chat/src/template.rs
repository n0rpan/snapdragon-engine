//! Chat template rendering with minijinja, made to give the same bytes as llama.cpp's own
//! jinja engine (`common/jinja`) for the Qwen3.x / Ornith templates:
//! - `tojson` / `string` / `safe` behave like llama.cpp's value.cpp (Python json.dumps
//!   separators `", "` / `": "`, no ASCII escaping, `True`/`False` for `bool|string`, Python
//!   float repr), and objects keep their insertion order;
//! - `raise_exception`, `strftime_now`, Python string/dict methods (pycompat);
//! - the context is built like `common_chat_template_direct_apply_impl`.

use minijinja::value::{Kwargs, Rest, Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind};
use serde_json::Value as Json;

/// Python `repr(float)` (shortest round-trip digits, `1.0`, `1e+20`, `1e-05`).
pub fn float_repr(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Rust's {:e} gives the shortest round-trip digits: d[.ddd]e<exp>
    let s = format!("{:e}", v.abs());
    let (mant, exp) = s.split_once('e').unwrap();
    let exp10: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let n = digits.len() as i32;
    let point = exp10 + 1;
    let mut out = String::new();
    if v.is_sign_negative() {
        out.push('-');
    }
    if point > -4 && point <= 16 {
        if point <= 0 {
            out += "0.";
            out += &"0".repeat((-point) as usize);
            out += &digits;
        } else if point >= n {
            out += &digits;
            out += &"0".repeat((point - n) as usize);
            out += ".0";
        } else {
            out += &digits[..point as usize];
            out.push('.');
            out += &digits[point as usize..];
        }
    } else {
        out += &digits[..1];
        if n > 1 {
            out.push('.');
            out += &digits[1..];
        }
        out += &format!("e{}{:02}", if exp10 < 0 { '-' } else { '+' }, exp10.abs());
    }
    out
}

fn float_json(v: f64) -> String {
    if v.is_nan() {
        "NaN".into()
    } else if v.is_infinite() {
        if v < 0.0 { "-Infinity".into() } else { "Infinity".into() }
    } else {
        float_repr(v)
    }
}

/// llama.cpp value_to_json string escaping.
pub fn json_escape_into(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Is this minijinja number a float (vs an integer)?
fn number_as_float(v: &Value) -> Option<f64> {
    let s = v.to_string();
    if s.contains(['.', 'e', 'E', 'n', 'N', 'i']) {
        s.parse::<f64>().ok().or_else(|| f64::try_from(v.clone()).ok())
    } else {
        None
    }
}

struct JsonFmt<'a> {
    indent: Option<usize>,
    item_sep: &'a str,
    key_sep: &'a str,
}

fn to_json(out: &mut String, v: &Value, lvl: usize, f: &JsonFmt) -> Result<(), Error> {
    if lvl > 100 {
        return Err(Error::new(ErrorKind::InvalidOperation, "value is nested too deep"));
    }
    let ind = |out: &mut String, l: usize| {
        if let Some(i) = f.indent {
            out.push_str(&" ".repeat(l * i));
        }
    };
    let nl = |out: &mut String| {
        if f.indent.is_some() {
            out.push('\n');
        }
    };
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
        ValueKind::Number => match number_as_float(v) {
            Some(x) => out.push_str(&float_json(x)),
            None => out.push_str(&v.to_string()),
        },
        ValueKind::String => json_escape_into(out, v.as_str().unwrap_or("")),
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = v.try_iter()?.collect();
            out.push('[');
            if !items.is_empty() {
                nl(out);
                for (i, it) in items.iter().enumerate() {
                    ind(out, lvl + 1);
                    to_json(out, it, lvl + 1, f)?;
                    if i + 1 < items.len() {
                        out.push_str(f.item_sep);
                    }
                    nl(out);
                }
                ind(out, lvl);
            }
            out.push(']');
        }
        ValueKind::Map => {
            let keys: Vec<Value> = v.try_iter()?.collect();
            out.push('{');
            if !keys.is_empty() {
                nl(out);
                for (i, k) in keys.iter().enumerate() {
                    ind(out, lvl + 1);
                    let ks = match k.kind() {
                        ValueKind::String => k.as_str().unwrap_or("").to_string(),
                        ValueKind::Number => match number_as_float(k) {
                            Some(x) => float_json(x),
                            None => k.to_string(),
                        },
                        _ => k.to_string(),
                    };
                    json_escape_into(out, &ks);
                    out.push_str(f.key_sep);
                    let item = v.get_item(k)?;
                    to_json(out, &item, lvl + 1, f)?;
                    if i + 1 < keys.len() {
                        out.push_str(f.item_sep);
                    }
                    nl(out);
                }
                ind(out, lvl);
            }
            out.push('}');
        }
        _ => out.push_str("null"),
    }
    Ok(())
}

/// `tojson(value, ensure_ascii=False, indent=None, separators=None)` like llama.cpp.
fn tojson_filter(v: Value, args: Rest<Value>, kwargs: Kwargs) -> Result<Value, Error> {
    let get = |name: &str, pos: usize| -> Result<Option<Value>, Error> {
        if let Some(x) = kwargs.get::<Option<Value>>(name)? {
            return Ok(Some(x));
        }
        Ok(args.get(pos).cloned())
    };
    let ensure_ascii = get("ensure_ascii", 0)?.map(|x| x.is_true()).unwrap_or(false);
    let indent = get("indent", 1)?.and_then(|x| i64::try_from(x).ok()).filter(|&i| i >= 0).map(|i| i as usize);
    let seps = get("separators", 2)?;
    if get("sort_keys", 3)?.map(|x| x.is_true()).unwrap_or(false) {
        return Err(Error::new(ErrorKind::InvalidOperation, "tojson sort_keys=true not implemented"));
    }
    kwargs.assert_all_used()?;
    let mut item_sep = if indent.is_none() { ", ".to_string() } else { ",".to_string() };
    let mut key_sep = ": ".to_string();
    if let Some(s) = seps {
        let v: Vec<Value> = s.try_iter()?.collect();
        if let Some(a) = v.first() {
            item_sep = a.to_string();
        }
        if let Some(b) = v.get(1) {
            key_sep = b.to_string();
        }
    }
    let mut out = String::new();
    to_json(&mut out, &v, 0, &JsonFmt { indent, item_sep: &item_sep, key_sep: &key_sep })?;
    if ensure_ascii {
        let mut o2 = String::with_capacity(out.len());
        for c in out.chars() {
            if (c as u32) < 0x80 {
                o2.push(c);
            } else {
                let mut b = [0u16; 2];
                for u in c.encode_utf16(&mut b) {
                    o2.push_str(&format!("\\u{:04x}", u));
                }
            }
        }
        out = o2;
    }
    Ok(Value::from_safe_string(out))
}

/// Python-ish `str()` of a value as llama.cpp's jinja does it (`string` and `safe` filters).
pub fn value_to_string(v: &Value) -> Result<String, Error> {
    Ok(match v.kind() {
        ValueKind::String => v.as_str().unwrap_or("").to_string(),
        ValueKind::Bool => if v.is_true() { "True" } else { "False" }.to_string(),
        ValueKind::None => "None".to_string(),
        ValueKind::Undefined => String::new(),
        ValueKind::Number => match number_as_float(v) {
            Some(x) => float_repr(x),
            None => v.to_string(),
        },
        _ => {
            let mut out = String::new();
            to_json(&mut out, v, 0, &JsonFmt { indent: None, item_sep: ", ", key_sep: ": " })?;
            out
        }
    })
}

fn string_filter(v: Value) -> Result<Value, Error> {
    Ok(Value::from_safe_string(value_to_string(&v)?))
}

fn raise_exception(msg: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, msg))
}

/// strftime with the local time zone (llama.cpp uses localtime too).
pub fn strftime_now(fmt: &str) -> String {
    let fmt_c = match std::ffi::CString::new(fmt) {
        Ok(c) => c,
        Err(_) => return String::new(),
    };
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        let mut buf = vec![0u8; 256];
        let n = libc::strftime(buf.as_mut_ptr() as *mut libc::c_char, buf.len(), fmt_c.as_ptr(), &tm);
        buf.truncate(n);
        String::from_utf8_lossy(&buf).into_owned()
    }
}

/// Which output format/parser goes with the template (`common_chat_template_detect_params_init`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Qwen3-Coder XML tool calls (Qwen3.5/3.6, Ornith, Qwen3-Coder, Nemotron Nano 3)
    Qwen3Coder,
    /// no tool-call syntax we know: plain content (+ <think> split if present)
    ContentOnly,
}

pub struct ChatTemplate {
    env: Environment<'static>,
    pub source: String,
    pub bos_token: String,
    pub eos_token: String,
    pub format: Format,
    /// template mentions `<think>`
    pub supports_thinking: bool,
}

impl ChatTemplate {
    pub fn new(source: &str, bos_token: &str, eos_token: &str) -> Result<ChatTemplate, Error> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson_filter);
        env.add_filter("string", string_filter);
        env.add_filter("safe", string_filter);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", |f: String| -> Result<Value, Error> { Ok(Value::from(strftime_now(&f))) });
        env.add_template_owned("chat", source.to_string())?;
        let format = if source.contains("<tool_call>")
            && source.contains("<function=")
            && source.contains("<parameter=")
            && !source.contains("'<tool_call><function=' ~ tool_call.name ~ '>'")
        {
            Format::Qwen3Coder
        } else {
            Format::ContentOnly
        };
        Ok(ChatTemplate {
            env,
            source: source.to_string(),
            bos_token: bos_token.to_string(),
            eos_token: eos_token.to_string(),
            format,
            supports_thinking: source.contains("<think>"),
        })
    }

    /// Render with an explicit context object (already normalized, see [`crate::request`]).
    pub fn render_ctx(&self, ctx: &Json) -> Result<String, Error> {
        let t = self.env.get_template("chat")?;
        let mut v = ctx.clone();
        if let Json::Object(m) = &mut v {
            m.insert("bos_token".into(), Json::String(self.bos_token.clone()));
            m.insert("eos_token".into(), Json::String(self.eos_token.clone()));
        }
        t.render(Value::from(minijinja::value::Serde(&v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn float_repr_python() {
        assert_eq!(float_repr(1.0), "1.0");
        assert_eq!(float_repr(0.1), "0.1");
        assert_eq!(float_repr(2.5e-3), "0.0025");
        assert_eq!(float_repr(1e-5), "1e-05");
        assert_eq!(float_repr(1e16), "1e+16");
        assert_eq!(float_repr(1e15), "1000000000000000.0");
        assert_eq!(float_repr(-3.25), "-3.25");
        assert_eq!(float_repr(123456.789), "123456.789");
    }
    #[test]
    fn tojson_like_llama() {
        let t = ChatTemplate::new(
            "{{ x | tojson }}|{{ b | string }}|{{ n | string }}|{{ f | string }}|{{ x | tojson(indent=2) }}",
            "",
            "",
        )
        .unwrap();
        let ctx = serde_json::json!({"x": {"z": 1, "a": [1.0, "é\n\"", null, true]}, "b": true, "n": 5, "f": 2.0});
        assert_eq!(
            t.render_ctx(&ctx).unwrap(),
            "{\"z\": 1, \"a\": [1.0, \"é\\n\\\"\", null, true]}|True|5|2.0|{\n  \"z\": 1,\n  \"a\": [\n    1.0,\n    \"é\\n\\\"\",\n    null,\n    true\n  ]\n}"
        );
    }
}
