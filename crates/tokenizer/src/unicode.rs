//! Port of the parts of llama.cpp `src/unicode.cpp` that the Qwen BPE path uses:
//! codepoint flags, `unicode_tolower`, the GPT-2 byte<->unicode map and the custom
//! qwen2/qwen35 pre-tokenizer splitters. Tables are generated from llama.cpp's
//! `unicode-data.cpp` (see tools/gen_unicode.py), so category answers are identical.

use crate::unicode_data::{MAP_LOWERCASE, RANGES_FLAGS, WHITESPACE};
use std::sync::OnceLock;

pub const UNDEFINED: u16 = 0x0001;
pub const NUMBER: u16 = 0x0002;
pub const LETTER: u16 = 0x0004;
pub const SEPARATOR: u16 = 0x0008;
pub const ACCENT_MARK: u16 = 0x0010;
pub const PUNCTUATION: u16 = 0x0020;
pub const SYMBOL: u16 = 0x0040;
pub const CONTROL: u16 = 0x0080;
pub const WHITESPACE_F: u16 = 0x0100;

const MAX_CODEPOINTS: usize = 0x110000;

fn flags_table() -> &'static [u16] {
    static T: OnceLock<Vec<u16>> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = vec![UNDEFINED; MAX_CODEPOINTS];
        for w in RANGES_FLAGS.windows(2) {
            let (ini, fl) = w[0];
            let end = w[1].0;
            for c in ini..end {
                t[c as usize] = fl;
            }
        }
        for &c in WHITESPACE.iter() {
            t[c as usize] |= WHITESPACE_F;
        }
        t
    })
}

/// Flags of a codepoint; `u32::MAX` (out of range sentinel) gives 0 like llama.cpp's
/// default-constructed `unicode_cpt_flags{}`.
#[inline]
pub fn flags(cpt: u32) -> u16 {
    let t = flags_table();
    if (cpt as usize) < t.len() {
        t[cpt as usize]
    } else {
        // llama.cpp returns UNDEFINED for cpt >= table size, but callers pass the
        // OUT_OF_RANGE sentinel only through `get_flags`, which returns 0 itself.
        UNDEFINED
    }
}

pub fn tolower(cpt: u32) -> u32 {
    match MAP_LOWERCASE.binary_search_by(|p| p.0.cmp(&cpt)) {
        Ok(i) => MAP_LOWERCASE[i].1,
        Err(_) => cpt,
    }
}

/// GPT-2 byte-level map: byte -> codepoint.
pub fn byte_to_cpt_table() -> &'static [char; 256] {
    static T: OnceLock<[char; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut m = ['\0'; 256];
        let mut set = [false; 256];
        for ch in (0x21..=0x7E).chain(0xA1..=0xAC).chain(0xAE..=0xFF) {
            m[ch] = char::from_u32(ch as u32).unwrap();
            set[ch] = true;
        }
        let mut n = 0u32;
        for ch in 0..256 {
            if !set[ch] {
                m[ch] = char::from_u32(256 + n).unwrap();
                n += 1;
            }
        }
        m
    })
}

/// Inverse of [`byte_to_cpt_table`]: codepoint -> byte.
pub fn cpt_to_byte(c: char) -> Option<u8> {
    static T: OnceLock<std::collections::HashMap<char, u8>> = OnceLock::new();
    let t = T.get_or_init(|| {
        byte_to_cpt_table().iter().enumerate().map(|(b, &c)| (c, b as u8)).collect()
    });
    t.get(&c).copied()
}

const OUT_OF_RANGE: u32 = 0xFFFF_FFFF;

/// Port of `unicode_regex_split_custom_qwen2` (marks=false) and
/// `unicode_regex_split_custom_qwen35` (marks=true). Returns word lengths in codepoints.
pub fn split_qwen(cpts: &[u32], marks: bool) -> Vec<usize> {
    let mut out = Vec::with_capacity(cpts.len() / 3 + 1);
    let end = cpts.len();
    let get_cpt = |p: usize| if p < end { cpts[p] } else { OUT_OF_RANGE };
    let get_flags = |p: usize| if p < end { flags(cpts[p]) } else { 0u16 };
    let is = |f: u16, bit: u16| f & bit != 0;
    // "letter" in the qwen35 sense is [\p{L}\p{M}]; qwen2 only \p{L}
    let lm = if marks { LETTER | ACCENT_MARK } else { LETTER };

    let mut prev_end = 0usize;
    let mut add = |e: usize, out: &mut Vec<usize>| -> usize {
        let len = e - prev_end;
        if len > 0 {
            out.push(len);
        }
        prev_end = e;
        len
    };

    let mut pos = 0usize;
    while pos < end {
        let cpt = get_cpt(pos);
        let fl = get_flags(pos);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if cpt == '\'' as u32 && pos + 1 < end {
            let n1 = tolower(get_cpt(pos + 1));
            if n1 == 's' as u32 || n1 == 't' as u32 || n1 == 'm' as u32 || n1 == 'd' as u32 {
                pos += add(pos + 2, &mut out);
                continue;
            }
            if pos + 2 < end {
                let n2 = tolower(get_cpt(pos + 2));
                if (n1 == 'r' as u32 && n2 == 'e' as u32)
                    || (n1 == 'v' as u32 && n2 == 'e' as u32)
                    || (n1 == 'l' as u32 && n2 == 'l' as u32)
                {
                    pos += add(pos + 3, &mut out);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}(\p{M})]+
        if !(cpt == '\r' as u32 || cpt == '\n' as u32 || is(fl, NUMBER)) {
            let hit = if marks {
                is(fl, LETTER) || is(fl, ACCENT_MARK) || is(get_flags(pos + 1), ACCENT_MARK) || is(get_flags(pos + 1), LETTER)
            } else {
                is(fl, LETTER) || is(get_flags(pos + 1), LETTER)
            };
            if hit {
                pos += 1;
                while get_flags(pos) & lm != 0 {
                    pos += 1;
                }
                add(pos, &mut out);
                continue;
            }
        }

        // \p{N}
        if is(fl, NUMBER) {
            pos += 1;
            add(pos, &mut out);
            continue;
        }

        // <space>?[^\s\p{L}(\p{M})\p{N}]+[\r\n]*
        let stop = WHITESPACE_F | lm | NUMBER;
        let mut fl2 = if cpt == ' ' as u32 { get_flags(pos + 1) } else { fl };
        if fl2 & stop == 0 && fl != 0 {
            if cpt == ' ' as u32 {
                pos += 1;
            }
            while fl2 & stop == 0 && fl2 != 0 {
                pos += 1;
                fl2 = get_flags(pos);
            }
            let mut c2 = get_cpt(pos);
            while c2 == '\r' as u32 || c2 == '\n' as u32 {
                pos += 1;
                c2 = get_cpt(pos);
            }
            add(pos, &mut out);
            continue;
        }

        let mut nws = 0usize;
        let mut last_rn = 0usize;
        while is(get_flags(pos + nws), WHITESPACE_F) {
            let c2 = get_cpt(pos + nws);
            if c2 == '\r' as u32 || c2 == '\n' as u32 {
                last_rn = pos + nws + 1;
            }
            nws += 1;
        }
        // \s*[\r\n]+
        if last_rn > 0 {
            pos = last_rn;
            add(pos, &mut out);
            continue;
        }
        // \s+(?!\S)
        if nws > 1 && get_cpt(pos + nws) != OUT_OF_RANGE {
            pos += nws - 1;
            add(pos, &mut out);
            continue;
        }
        // \s+
        if nws > 0 {
            pos += nws;
            add(pos, &mut out);
            continue;
        }
        // no match
        pos += 1;
        add(pos, &mut out);
    }
    out
}
