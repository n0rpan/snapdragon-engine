//! Check64: the streaming 64-bit checksum of the state files (server `prefix_store`, `state_store`)
//! and of the kernel-path libm probe ([`crate::engine::kernel_path`]).

/// Streaming 64-bit checksum of an entry's state bytes (integrity against truncation and
/// bit rot, not an adversary: the directory is 0700 on the encrypted home). One multiply-xor-
/// rotate lane over little-endian u64 words, several GB/s, so ~60 ms for a 285 MB Flash state.
#[derive(Clone)]
pub struct Check64 {
    h: u64,
    tail: [u8; 8],
    nt: usize,
    pub len: u64,
}

impl Default for Check64 {
    fn default() -> Self {
        Check64 { h: 0x243f_6a88_85a3_08d3, tail: [0; 8], nt: 0, len: 0 }
    }
}

impl Check64 {
    #[inline]
    fn word(&mut self, w: u64) {
        self.h = (self.h ^ w).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29);
    }
    pub fn update(&mut self, mut d: &[u8]) {
        self.len += d.len() as u64;
        if self.nt > 0 {
            let n = (8 - self.nt).min(d.len());
            self.tail[self.nt..self.nt + n].copy_from_slice(&d[..n]);
            self.nt += n;
            d = &d[n..];
            if self.nt < 8 {
                return;
            }
            let w = u64::from_le_bytes(self.tail);
            self.word(w);
            self.nt = 0;
        }
        let mut it = d.chunks_exact(8);
        for c in &mut it {
            self.word(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let r = it.remainder();
        self.tail[..r.len()].copy_from_slice(r);
        self.nt = r.len();
    }
    pub fn finish(mut self) -> u64 {
        if self.nt > 0 {
            let mut t = [0u8; 8];
            t[..self.nt].copy_from_slice(&self.tail[..self.nt]);
            self.word(u64::from_le_bytes(t) ^ ((self.nt as u64) << 56));
        }
        let len = self.len;
        self.word(len);
        let mut h = self.h;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^ (h >> 33)
    }
}

