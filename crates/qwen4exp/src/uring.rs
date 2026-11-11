//! Minimal io_uring (raw syscalls, no dependency): one ring, IORING_OP_READ / READ_FIXED only, owned by a
//! single thread. By default buffers are not registered (RLIMIT_MEMLOCK is 8 MiB here; the kernel pins
//! the pages of each O_DIRECT read for its duration). `RingOpts` adds the optional setup flags the IO bench
//! sweeps (SQPOLL with a pinned kernel submit thread, DEFER_TASKRUN) and `register_file` / `register_buffer`
//! (fixed file, fixed buffer). This file has no crate-internal dependency: tools/flash-iobench includes it by path.

use std::io;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering::*};

const IORING_OFF_SQ_RING: i64 = 0;
const IORING_OFF_CQ_RING: i64 = 0x8000000;
const IORING_OFF_SQES: i64 = 0x10000000;
const IORING_FEAT_SINGLE_MMAP: u32 = 1;
const IORING_ENTER_GETEVENTS: u32 = 1;
const IORING_ENTER_SQ_WAKEUP: u32 = 2;
const IORING_OP_READ: u8 = 22;
const IORING_OP_READ_FIXED: u8 = 4;
const IORING_SETUP_SQPOLL: u32 = 1 << 1;
const IORING_SETUP_SQ_AFF: u32 = 1 << 2;
const IORING_SETUP_COOP_TASKRUN: u32 = 1 << 8;
const IORING_SETUP_SINGLE_ISSUER: u32 = 1 << 12;
const IORING_SETUP_DEFER_TASKRUN: u32 = 1 << 13;
const IORING_SQ_NEED_WAKEUP: u32 = 1;
const IORING_REGISTER_BUFFERS: u32 = 0;
const IORING_REGISTER_FILES: u32 = 2;
const IOSQE_FIXED_FILE: u8 = 1;

/// Optional ring setup (all off = the engine's historical ring).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RingOpts {
    /// IORING_SETUP_SQPOLL: a kernel thread submits; `Some(cpu)` also pins it (SQ_AFF)
    pub sqpoll: bool,
    pub sqpoll_cpu: Option<u32>,
    /// SQPOLL thread idle time before it sleeps, ms (0 = kernel default 1000)
    pub sqpoll_idle_ms: u32,
    /// COOP_TASKRUN | SINGLE_ISSUER | DEFER_TASKRUN (completions are posted inside io_uring_enter(GETEVENTS));
    /// ignored with SQPOLL (the kernel refuses the combination)
    pub defer_taskrun: bool,
}

#[repr(C)]
#[derive(Default)]
struct SqOff {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    flags: u32,
    dropped: u32,
    array: u32,
    resv1: u32,
    user_addr: u64,
}
#[repr(C)]
#[derive(Default)]
struct CqOff {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    overflow: u32,
    cqes: u32,
    flags: u32,
    resv1: u32,
    user_addr: u64,
}
#[repr(C)]
#[derive(Default)]
struct Params {
    sq_entries: u32,
    cq_entries: u32,
    flags: u32,
    sq_thread_cpu: u32,
    sq_thread_idle: u32,
    features: u32,
    wq_fd: u32,
    resv: [u32; 3],
    sq_off: SqOff,
    cq_off: CqOff,
}
const _: () = assert!(std::mem::size_of::<Params>() == 120);

#[repr(C)]
struct Sqe {
    opcode: u8,
    flags: u8,
    ioprio: u16,
    fd: i32,
    off: u64,
    addr: u64,
    len: u32,
    rw_flags: u32,
    user_data: u64,
    buf_index: u16,
    personality: u16,
    splice_fd_in: i32,
    addr3: u64,
    pad: u64,
}
const _: () = assert!(std::mem::size_of::<Sqe>() == 64);

#[repr(C)]
struct Cqe {
    user_data: u64,
    res: i32,
    flags: u32,
}

pub struct Ring {
    fd: i32,
    sq_ptr: *mut u8,
    sq_len: usize,
    cq_ptr: *mut u8,
    cq_len: usize,
    sqes: *mut Sqe,
    sqes_len: usize,
    sq_head: *const AtomicU32,
    sq_tail: *const AtomicU32,
    sq_mask: u32,
    sq_entries: u32,
    sq_array: *mut u32,
    sq_flags: *const AtomicU32,
    sqpoll: bool,
    /// a file / buffer is registered (index 0)
    fixed_file: Option<i32>,
    fixed_buf: Option<(usize, usize)>,
    cq_head: *const AtomicU32,
    cq_tail: *const AtomicU32,
    cq_mask: u32,
    cqes: *const Cqe,
    /// SQEs written but not yet passed to io_uring_enter
    unsubmitted: u32,
    pub inflight: u32,
}

// The ring is used by exactly one thread at a time (moved into the IO thread).
unsafe impl Send for Ring {}

pub struct Completion {
    pub user_data: u64,
    pub res: i32,
}

impl Ring {
    pub fn new(entries: u32) -> io::Result<Ring> {
        Ring::with_opts(entries, RingOpts::default())
    }

    pub fn with_opts(entries: u32, o: RingOpts) -> io::Result<Ring> {
        let mut p = Params::default();
        if o.sqpoll {
            p.flags |= IORING_SETUP_SQPOLL;
            p.sq_thread_idle = o.sqpoll_idle_ms;
            if let Some(c) = o.sqpoll_cpu {
                p.flags |= IORING_SETUP_SQ_AFF;
                p.sq_thread_cpu = c;
            }
        } else if o.defer_taskrun {
            p.flags |= IORING_SETUP_COOP_TASKRUN | IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN;
        }
        // SAFETY: plain syscall with a correctly sized params struct.
        let fd = unsafe { libc::syscall(libc::SYS_io_uring_setup, entries, &mut p as *mut Params) } as i32;
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let sq_len = p.sq_off.array as usize + p.sq_entries as usize * 4;
        let cq_len = p.cq_off.cqes as usize + p.cq_entries as usize * std::mem::size_of::<Cqe>();
        let single = p.features & IORING_FEAT_SINGLE_MMAP != 0;
        let map = |len: usize, off: i64| -> io::Result<*mut u8> {
            // SAFETY: mapping the ring regions the kernel documented for this fd.
            let r = unsafe {
                libc::mmap(ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED | libc::MAP_POPULATE, fd, off)
            };
            if r == libc::MAP_FAILED {
                Err(io::Error::last_os_error())
            } else {
                Ok(r as *mut u8)
            }
        };
        let (sq_len, cq_len) = if single { (sq_len.max(cq_len), sq_len.max(cq_len)) } else { (sq_len, cq_len) };
        let sq_ptr = map(sq_len, IORING_OFF_SQ_RING)?;
        let cq_ptr = if single { sq_ptr } else { map(cq_len, IORING_OFF_CQ_RING)? };
        let sqes_len = p.sq_entries as usize * 64;
        let sqes = map(sqes_len, IORING_OFF_SQES)? as *mut Sqe;
        // SAFETY: offsets come from the kernel for these mappings.
        unsafe {
            Ok(Ring {
                fd,
                sq_ptr,
                sq_len,
                cq_ptr,
                cq_len: if single { 0 } else { cq_len },
                sqes,
                sqes_len,
                sq_head: sq_ptr.add(p.sq_off.head as usize) as *const AtomicU32,
                sq_tail: sq_ptr.add(p.sq_off.tail as usize) as *const AtomicU32,
                sq_mask: *(sq_ptr.add(p.sq_off.ring_mask as usize) as *const u32),
                sq_entries: p.sq_entries,
                sq_array: sq_ptr.add(p.sq_off.array as usize) as *mut u32,
                sq_flags: sq_ptr.add(p.sq_off.flags as usize) as *const AtomicU32,
                sqpoll: o.sqpoll,
                fixed_file: None,
                fixed_buf: None,
                cq_head: cq_ptr.add(p.cq_off.head as usize) as *const AtomicU32,
                cq_tail: cq_ptr.add(p.cq_off.tail as usize) as *const AtomicU32,
                cq_mask: *(cq_ptr.add(p.cq_off.ring_mask as usize) as *const u32),
                cqes: cq_ptr.add(p.cq_off.cqes as usize) as *const Cqe,
                unsubmitted: 0,
                inflight: 0,
            })
        }
    }

    pub fn sq_space(&self) -> u32 {
        // SAFETY: ring words are valid for the ring's lifetime.
        let head = unsafe { (*self.sq_head).load(Acquire) };
        let tail = unsafe { (*self.sq_tail).load(Relaxed) };
        self.sq_entries - tail.wrapping_sub(head)
    }

    /// Register `fd` as fixed file 0: later reads of this fd skip the per-request file reference.
    pub fn register_file(&mut self, fd: i32) -> io::Result<()> {
        let fds = [fd];
        // SAFETY: plain syscall, one-element fd array.
        let r = unsafe { libc::syscall(libc::SYS_io_uring_register, self.fd, IORING_REGISTER_FILES, fds.as_ptr(), 1u32) };
        if r < 0 {
            return Err(io::Error::last_os_error());
        }
        self.fixed_file = Some(fd);
        Ok(())
    }

    /// Register `[ptr, ptr + len)` as fixed buffer 0 (pages pinned once; counts against RLIMIT_MEMLOCK without
    /// CAP_IPC_LOCK): reads whose buffer lies inside use READ_FIXED.
    ///
    /// # Safety
    /// The region must stay mapped for the ring's lifetime.
    pub unsafe fn register_buffer(&mut self, ptr: *mut u8, len: usize) -> io::Result<()> {
        let iov = [libc::iovec { iov_base: ptr as *mut libc::c_void, iov_len: len }];
        let r = libc::syscall(libc::SYS_io_uring_register, self.fd, IORING_REGISTER_BUFFERS, iov.as_ptr(), 1u32);
        if r < 0 {
            return Err(io::Error::last_os_error());
        }
        self.fixed_buf = Some((ptr as usize, len));
        Ok(())
    }

    /// Queue one read of `len` bytes at file offset `off` into `buf`. Returns false if the SQ is full.
    ///
    /// # Safety
    /// `buf` must stay valid and untouched until the completion with `user_data` is reaped.
    pub unsafe fn push_read(&mut self, fd: i32, buf: *mut u8, len: u32, off: u64, user_data: u64) -> bool {
        if self.sq_space() == 0 {
            return false;
        }
        let tail = (*self.sq_tail).load(Relaxed);
        let idx = tail & self.sq_mask;
        let sqe = self.sqes.add(idx as usize);
        let fixed_buf = self.fixed_buf.is_some_and(|(b, l)| buf as usize >= b && buf as usize + len as usize <= b + l);
        let fixed_file = self.fixed_file == Some(fd);
        ptr::write(
            sqe,
            Sqe {
                opcode: if fixed_buf { IORING_OP_READ_FIXED } else { IORING_OP_READ },
                flags: if fixed_file { IOSQE_FIXED_FILE } else { 0 },
                ioprio: 0,
                fd: if fixed_file { 0 } else { fd },
                off,
                addr: buf as u64,
                len,
                rw_flags: 0,
                user_data,
                buf_index: 0,
                personality: 0,
                splice_fd_in: 0,
                addr3: 0,
                pad: 0,
            },
        );
        *self.sq_array.add(idx as usize) = idx;
        (*self.sq_tail).store(tail.wrapping_add(1), Release);
        self.unsubmitted += 1;
        self.inflight += 1;
        true
    }

    /// Submit queued SQEs; with `wait`, block until at least one completion is available.
    pub fn enter(&mut self, wait: bool) -> io::Result<()> {
        if self.sqpoll {
            // the kernel thread consumes the SQ by itself; wake it if it went to sleep
            self.unsubmitted = 0;
            // SAFETY: ring word valid for the ring's lifetime.
            let need = unsafe { (*self.sq_flags).load(Acquire) } & IORING_SQ_NEED_WAKEUP != 0;
            if need || wait {
                let flags = if need { IORING_ENTER_SQ_WAKEUP } else { 0 } | if wait { IORING_ENTER_GETEVENTS } else { 0 };
                // SAFETY: plain syscall on our ring fd.
                let r = unsafe { libc::syscall(libc::SYS_io_uring_enter, self.fd, 0u32, if wait { 1u32 } else { 0 }, flags, ptr::null::<u8>(), 0usize) };
                if r < 0 {
                    let e = io::Error::last_os_error();
                    if e.raw_os_error() != Some(libc::EINTR) && e.raw_os_error() != Some(libc::EBUSY) {
                        return Err(e);
                    }
                }
            }
            return Ok(());
        }
        let flags = if wait || self.inflight > 0 { IORING_ENTER_GETEVENTS } else { 0 };
        let min = if wait { 1 } else { 0 };
        if self.unsubmitted == 0 && flags == 0 {
            return Ok(());
        }
        // SAFETY: plain syscall on our ring fd.
        let r = unsafe {
            libc::syscall(libc::SYS_io_uring_enter, self.fd, self.unsubmitted, min, flags, ptr::null::<u8>(), 0usize)
        };
        if r < 0 {
            let e = io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::EINTR) || e.raw_os_error() == Some(libc::EBUSY) {
                return Ok(());
            }
            return Err(e);
        }
        self.unsubmitted -= (r as u32).min(self.unsubmitted);
        Ok(())
    }

    /// Reap every available completion.
    pub fn reap(&mut self, mut f: impl FnMut(Completion)) -> u32 {
        let mut n = 0;
        // SAFETY: ring words / CQE array are valid for the ring's lifetime.
        unsafe {
            let mut head = (*self.cq_head).load(Relaxed);
            let tail = (*self.cq_tail).load(Acquire);
            while head != tail {
                let c = &*self.cqes.add((head & self.cq_mask) as usize);
                f(Completion { user_data: c.user_data, res: c.res });
                head = head.wrapping_add(1);
                n += 1;
            }
            (*self.cq_head).store(head, Release);
        }
        self.inflight -= n;
        n
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        // SAFETY: unmapping our own mappings, closing our own fd.
        unsafe {
            libc::munmap(self.sqes as *mut libc::c_void, self.sqes_len);
            if self.cq_len > 0 {
                libc::munmap(self.cq_ptr as *mut libc::c_void, self.cq_len);
            }
            libc::munmap(self.sq_ptr as *mut libc::c_void, self.sq_len);
            libc::close(self.fd);
        }
    }
}

/// Anonymous, page-aligned, THP-advised memory region (never freed until drop).
pub struct Arena {
    pub ptr: *mut u8,
    pub len: usize,
}
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    pub fn new(len: usize, prefault: bool) -> io::Result<Arena> {
        const HUGE: usize = 2 << 20;
        let len = len.max(4096).div_ceil(HUGE) * HUGE;
        // FD-5: map len + 2 MB and trim both ends so the arena starts on a 2 MB boundary: every 2 MB of it can be a
        // transparent huge page (mmap only guarantees 4 KiB alignment, which left the first/last partial 2 MB as 4 KiB pages)
        // SAFETY: anonymous private mapping.
        let raw = unsafe {
            libc::mmap(ptr::null_mut(), len + HUGE, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        if raw == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        // FLASH_ARENA_ALIGN2M=1 arms the alignment (FD-5 ABBA arm); default keeps the old placement (mmap's own address)
        let align = std::env::var("FLASH_ARENA_ALIGN2M").map(|v| v == "1").unwrap_or(false);
        let start = if align { (raw as usize).div_ceil(HUGE) * HUGE } else { raw as usize };
        let head = start - raw as usize;
        let tail = HUGE - head;
        // SAFETY: trimming the unused ends of our own mapping.
        unsafe {
            if head > 0 {
                libc::munmap(raw, head);
            }
            if tail > 0 {
                libc::munmap((start + len) as *mut libc::c_void, tail);
            }
        }
        let p = start as *mut libc::c_void;
        // SAFETY: advisory on our mapping.
        unsafe { libc::madvise(p, len, libc::MADV_HUGEPAGE) };
        let a = Arena { ptr: p as *mut u8, len };
        if prefault {
            // touch every 4 KiB page so the timing loop never page-faults
            for off in (0..len).step_by(4096) {
                // SAFETY: inside the mapping.
                unsafe { ptr::write_volatile(a.ptr.add(off), 0) };
            }
        }
        Ok(a)
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: our mapping.
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
    }
}

pub fn open_direct(path: &std::path::Path) -> io::Result<i32> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: plain open.
    let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDONLY | libc::O_DIRECT | libc::O_CLOEXEC) };
    if fd < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(fd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn ring_reads_a_scratch_file_with_o_direct() {
        let dir = std::env::temp_dir().join(format!("oxf-uring-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("f.bin");
        let data: Vec<u8> = (0..(4 << 20)).map(|i: u32| (i.wrapping_mul(2654435761) >> 24) as u8).collect();
        std::fs::File::create(&p).unwrap().write_all(&data).unwrap();
        let fd = match open_direct(&p) {
            Ok(fd) => fd,
            Err(e) => {
                eprintln!("O_DIRECT unsupported here ({e}); skipping");
                return;
            }
        };
        let mut ring = Ring::new(32).unwrap();
        let arena = Arena::new(4 << 20, false).unwrap();
        unsafe {
            for i in 0..4u64 {
                assert!(ring.push_read(fd, arena.ptr.add((i as usize) << 20), 1 << 20, (3 - i) << 20, i));
            }
        }
        let mut done = 0;
        while done < 4 {
            ring.enter(true).unwrap();
            ring.reap(|c| {
                assert_eq!(c.res, 1 << 20, "read {} failed: {}", c.user_data, c.res);
                done += 1;
            });
        }
        for i in 0..4usize {
            let got = unsafe { std::slice::from_raw_parts(arena.ptr.add(i << 20), 1 << 20) };
            assert_eq!(got, &data[(3 - i) << 20..(4 - i) << 20]);
        }
        unsafe { libc::close(fd) };
        let _ = std::fs::remove_dir_all(&dir);
    }
}
