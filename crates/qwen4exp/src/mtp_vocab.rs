//! Corpus token-frequency counting for picking a restricted vocabulary for the MTP draft head.

use std::fs;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"MTPV1\0\0\0";
const SKIP_DIRS: [&str; 5] = [".git", "target", "node_modules", ".venv", "build"];

/// Token counts over all files (lossy UTF-8, each file read whole). Files that are larger than
/// `max_file_bytes` or unreadable are skipped. Returns (counts indexed by id, total counted
/// tokens, number of files used).
pub fn count_files(
    tk: &tokenizer::Tokenizer,
    files: &[PathBuf],
    n_vocab: usize,
    max_file_bytes: u64,
) -> (Vec<u64>, u64, usize) {
    let mut counts = vec![0u64; n_vocab];
    let mut total = 0u64;
    let mut used = 0usize;
    for f in files {
        let len = match fs::metadata(f) {
            Ok(m) => m.len(),
            Err(_) => continue,
        };
        if len > max_file_bytes {
            continue;
        }
        let bytes = match fs::read(f) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let text = String::from_utf8_lossy(&bytes);
        used += 1;
        for t in tk.encode(&text, false, false) {
            let i = t as usize;
            if i < n_vocab {
                counts[i] += 1;
                total += 1;
            }
        }
    }
    (counts, total, used)
}

/// Top-n ids by count (ties: lower id first). The ids in `always` (negative or out-of-range
/// ids are ignored) take priority and count toward n; if there are more of them than n, the
/// first n in the given order are kept. Result is sorted ascending, length min(n, counts.len()).
pub fn top_ids(counts: &[u64], n: usize, always: &[i32]) -> Vec<i32> {
    let k = n.min(counts.len());
    let mut taken = vec![false; counts.len()];
    let mut out: Vec<i32> = Vec::with_capacity(k);
    for &a in always {
        if out.len() >= k {
            break;
        }
        if a < 0 {
            continue;
        }
        let i = a as usize;
        if i < counts.len() && !taken[i] {
            taken[i] = true;
            out.push(a);
        }
    }
    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by(|&x, &y| counts[y].cmp(&counts[x]).then(x.cmp(&y)));
    for i in order {
        if out.len() >= k {
            break;
        }
        if !taken[i] {
            taken[i] = true;
            out.push(i as i32);
        }
    }
    out.sort_unstable();
    out
}

/// Fraction of the corpus tokens covered by `ids` (duplicates and out-of-range ids ignored).
/// Returns 0.0 when the corpus is empty.
pub fn coverage(counts: &[u64], ids: &[i32]) -> f64 {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let mut seen = vec![false; counts.len()];
    let mut covered = 0u64;
    for &id in ids {
        if id < 0 {
            continue;
        }
        let i = id as usize;
        if i < counts.len() && !seen[i] {
            seen[i] = true;
            covered += counts[i];
        }
    }
    covered as f64 / total as f64
}

/// FNV-1a 64-bit over `bytes`. Stable across runs and platforms (unlike std's hashers).
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

/// Held-out test for a file: true iff fnv1a64(path string) % k == 0. `k == 0` never holds out.
pub fn is_holdout(path: &Path, k: usize) -> bool {
    k > 0 && fnv1a64(path.to_string_lossy().as_bytes()) % (k as u64) == 0
}

/// Splits `files` into (train, held-out) by `is_holdout`, keeping the input order.
pub fn split_holdout(files: &[PathBuf], k: usize) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut train = Vec::new();
    let mut hold = Vec::new();
    for f in files {
        if is_holdout(f, k) {
            hold.push(f.clone());
        } else {
            train.push(f.clone());
        }
    }
    (train, hold)
}

/// Number of distinct ids with a nonzero count that are not in `ids`.
pub fn distinct_missing(counts: &[u64], ids: &[i32]) -> usize {
    let mut listed = vec![false; counts.len()];
    for &id in ids {
        if id >= 0 && (id as usize) < counts.len() {
            listed[id as usize] = true;
        }
    }
    let mut n = 0usize;
    for (i, &c) in counts.iter().enumerate() {
        if c > 0 && !listed[i] {
            n += 1;
        }
    }
    n
}

/// File format: magic b"MTPV1\0\0\0", u64 LE count, then count i32 LE ids.
pub fn write_ids(path: &Path, ids: &[i32]) -> Result<(), String> {
    let mut buf: Vec<u8> = Vec::with_capacity(16 + 4 * ids.len());
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&(ids.len() as u64).to_le_bytes());
    for id in ids {
        buf.extend_from_slice(&id.to_le_bytes());
    }
    fs::write(path, buf).map_err(|e| format!("write {}: {e}", path.display()))
}

pub fn read_ids(path: &Path) -> Result<Vec<i32>, String> {
    let buf = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if buf.len() < 16 || &buf[..8] != &MAGIC[..] {
        return Err(format!("{}: not an MTPV1 id file", path.display()));
    }
    let mut cnt = [0u8; 8];
    cnt.copy_from_slice(&buf[8..16]);
    let count = u64::from_le_bytes(cnt);
    let body = &buf[16..];
    let expect = count
        .checked_mul(4)
        .ok_or_else(|| format!("{}: bad count {count}", path.display()))?;
    if body.len() as u64 != expect {
        return Err(format!(
            "{}: count {count} does not match {} payload bytes",
            path.display(),
            body.len()
        ));
    }
    Ok(body
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Recursive file walk: regular files whose extension is in `exts` (no dot, compared
/// lowercase). Directories named .git, target, node_modules, .venv or build are skipped;
/// symlinks are not followed. Stops after `max_files`. Each directory is visited in sorted order.
pub fn walk(roots: &[PathBuf], exts: &[&str], max_files: usize) -> Vec<PathBuf> {
    let exts: Vec<String> = exts.iter().map(|e| e.to_ascii_lowercase()).collect();
    let mut out: Vec<PathBuf> = Vec::new();
    for r in roots {
        if out.len() >= max_files {
            break;
        }
        match fs::metadata(r) {
            Ok(m) if m.is_dir() => walk_dir(r, &exts, max_files, &mut out),
            Ok(m) if m.is_file() && has_ext(r, &exts) => out.push(r.clone()),
            _ => {}
        }
    }
    out
}

fn walk_dir(dir: &Path, exts: &[String], max_files: usize, out: &mut Vec<PathBuf>) {
    let rd = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return,
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if out.len() >= max_files {
            return;
        }
        let ft = match fs::symlink_metadata(&p) {
            Ok(m) => m.file_type(),
            Err(_) => continue,
        };
        if ft.is_dir() {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if !SKIP_DIRS.contains(&name) {
                walk_dir(&p, exts, max_files, out);
            }
        } else if ft.is_file() && has_ext(&p, exts) {
            out.push(p);
        }
    }
}

fn has_ext(p: &Path, exts: &[String]) -> bool {
    match p.extension().and_then(|s| s.to_str()) {
        Some(e) => {
            let e = e.to_ascii_lowercase();
            exts.iter().any(|x| *x == e)
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COUNTS: [u64; 6] = [5, 9, 9, 0, 2, 9];

    #[test]
    fn top_ids_ties_lower_id_first() {
        // count order: 1,2,5 (9 each; ties by id), 0 (5), 4 (2), 3 (0)
        assert_eq!(top_ids(&COUNTS, 3, &[]), vec![1, 2, 5]);
        assert_eq!(top_ids(&COUNTS, 4, &[]), vec![0, 1, 2, 5]);
        assert_eq!(top_ids(&COUNTS, 2, &[]), vec![1, 2]);
    }

    #[test]
    fn top_ids_always_is_included() {
        // id 0 is forced in and takes one of the three slots
        assert_eq!(top_ids(&COUNTS, 3, &[0]), vec![0, 1, 2]);
        // already in the top set: no duplicate, no extra slot used
        assert_eq!(top_ids(&COUNTS, 3, &[5]), vec![1, 2, 5]);
        // out-of-range and negative ids are ignored
        assert_eq!(top_ids(&COUNTS, 2, &[-1, 99]), vec![1, 2]);
    }

    #[test]
    fn top_ids_fills_with_lowest_unused_ids() {
        let counts = [0u64, 3, 0, 0, 1];
        // 1 (3), 4 (1), then zero-count ids in ascending order: 0, 2
        assert_eq!(top_ids(&counts, 4, &[]), vec![0, 1, 2, 4]);
    }

    #[test]
    fn top_ids_length_is_capped_by_vocab() {
        assert_eq!(top_ids(&COUNTS, 10, &[]).len(), COUNTS.len());
        assert!(top_ids(&COUNTS, 0, &[]).is_empty());
    }

    #[test]
    fn coverage_fraction() {
        let total = COUNTS.iter().sum::<u64>() as f64;
        let c = coverage(&COUNTS, &[1, 2]);
        assert!((c - (COUNTS[1] + COUNTS[2]) as f64 / total).abs() < 1e-12, "got {c}");
        // duplicates are not counted twice
        let c = coverage(&COUNTS, &[1, 1]);
        assert!((c - COUNTS[1] as f64 / total).abs() < 1e-12, "got {c}");
        assert!((coverage(&COUNTS, &[]) - 0.0).abs() < 1e-12);
        assert!((coverage(&COUNTS, &[0, 1, 2, 3, 4, 5]) - 1.0).abs() < 1e-12);
        assert!((coverage(&[0u64; 4], &[1]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn write_read_round_trip() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "mtp_vocab_rt_{}_{}.bin",
            std::process::id(),
            nanos
        ));
        let ids = vec![0, 7, 248046, 248044];
        write_ids(&path, &ids).unwrap();
        let back = read_ids(&path);
        let _ = fs::remove_file(&path);
        assert_eq!(back.unwrap(), ids);
    }

    #[test]
    fn read_rejects_bad_magic() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "mtp_vocab_bad_{}_{}.bin",
            std::process::id(),
            nanos
        ));
        fs::write(&path, b"NOTMTPV1xxxxxxxx").unwrap();
        let r = read_ids(&path);
        let _ = fs::remove_file(&path);
        assert!(r.is_err());
    }

    #[test]
    fn fnv1a64_known_values() {
        // offset basis for the empty string; "a" is 0x61, one FNV-1a round from the basis
        assert_eq!(fnv1a64(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a64(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a64(b"a"), fnv1a64("a".as_bytes()));
    }

    #[test]
    fn holdout_split_partitions_files() {
        let files: Vec<PathBuf> = (0..200).map(|i| PathBuf::from(format!("src/f{i}.rs"))).collect();
        let (train, hold) = split_holdout(&files, 5);
        assert_eq!(train.len() + hold.len(), files.len());
        assert!(!hold.is_empty() && !train.is_empty());
        for f in &hold {
            assert!(is_holdout(f, 5));
        }
        for f in &train {
            assert!(!is_holdout(f, 5));
        }
        // every file lands in exactly one set
        for f in &files {
            let n = train.iter().filter(|t| *t == f).count() + hold.iter().filter(|h| *h == f).count();
            assert_eq!(n, 1, "{}", f.display());
        }
    }

    #[test]
    fn holdout_zero_keeps_everything_in_train() {
        let files: Vec<PathBuf> = (0..50).map(|i| PathBuf::from(format!("d/{i}.md"))).collect();
        let (train, hold) = split_holdout(&files, 0);
        assert_eq!(train, files);
        assert!(hold.is_empty());
    }

    #[test]
    fn holdout_coverage_arithmetic() {
        // list is built from the train counts; held-out counts are scored against it
        let train = [4u64, 0, 3, 1];
        let hold = [1u64, 2, 0, 1];
        let ids = top_ids(&train, 2, &[]); // id 0 (4), id 2 (3)
        assert_eq!(ids, vec![0, 2]);
        assert!((coverage(&train, &ids) - 7.0 / 8.0).abs() < 1e-12);
        assert!((coverage(&hold, &ids) - 1.0 / 4.0).abs() < 1e-12);
        // held-out ids not in the list: 1 (2 tokens) and 3 (1 token)
        assert_eq!(distinct_missing(&hold, &ids), 2);
        // train ids not in the list: only 3
        assert_eq!(distinct_missing(&train, &ids), 1);
        assert!((coverage(&[0u64; 3], &ids) - 0.0).abs() < 1e-12);
    }
}
