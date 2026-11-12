//! CPU allocations owned independently of DSP sessions. Sealing makes weights
//! immutable; imports retain a clone until their DSP/driver mappings are gone.
use std::fmt;
use std::os::fd::{AsFd, BorrowedFd, RawFd};
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc,
};

pub(crate) struct RawMemory {
    pub ptr: *mut u8,
    pub len: usize,
    pub fd: RawFd,
    pub token: usize,
}
// SAFETY: implementations allocate exclusive, zeroed, aligned CPU memory with
// a valid borrowed fd, retaining all allocator state through release. Release
// runs once per successful allocation. Only sealed/read-only buffers are shared.
pub(crate) unsafe trait MemoryPool: Send + Sync {
    fn allocate(&self, bytes: usize) -> Result<RawMemory, String>;
    unsafe fn release(&self, raw: &RawMemory);
}
struct Ledger {
    owned: AtomicUsize,
    limit: usize,
}
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub struct SharedAllocator {
    pool: Arc<dyn MemoryPool>,
    ledger: Arc<Ledger>,
}
impl SharedAllocator {
    pub(crate) fn from_pool(pool: Arc<dyn MemoryPool>, limit: usize) -> Result<Self, String> {
        if limit == 0 || limit % 4096 != 0 {
            return Err("shared allocator: nonzero4KiB budget required".into());
        }
        Ok(Self {
            pool,
            ledger: Arc::new(Ledger {
                owned: AtomicUsize::new(0),
                limit,
            }),
        })
    }
    /// Loads only the pinned runtime allocator; no DSP session or mapping.
    #[cfg(feature = "device")]
    pub fn open(limit: usize) -> Result<Self, String> {
        Self::from_pool(Arc::new(crate::device::CpuMemoryPool::open()?), limit)
    }
    /// Bytes still OWNED through live allocation references, not a measurement
    /// of physical residency or deferred driver release. Imports don't double it.
    pub fn owned_bytes(&self) -> usize {
        self.ledger.owned.load(Ordering::Acquire)
    }
    pub fn allocate(&self, bytes: usize) -> Result<MutableMemory, String> {
        if bytes == 0 || bytes % 4096 != 0 || bytes > 512 << 20 {
            return Err("shared allocation:4KiB multiples <=512MiB required".into());
        }
        let id = NEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| "shared allocation identity exhausted")?;
        self.ledger
            .owned
            .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |v| {
                v.checked_add(bytes).filter(|&n| n <= self.ledger.limit)
            })
            .map_err(|_| "shared allocation budget exceeded")?;
        let raw = match self.pool.allocate(bytes) {
            Ok(v) => v,
            Err(e) => {
                self.ledger.owned.fetch_sub(bytes, Ordering::AcqRel);
                return Err(e);
            }
        };
        if raw.ptr.is_null() || raw.ptr as usize % 4096 != 0 || raw.len != bytes || raw.fd < 0 {
            // SAFETY: a successful pool allocation owns raw until this release.
            unsafe { self.pool.release(&raw) };
            self.ledger.owned.fetch_sub(bytes, Ordering::AcqRel);
            return Err("shared allocation: invalid allocator result".into());
        }
        Ok(MutableMemory {
            inner: Allocation {
                id,
                raw,
                pool: self.pool.clone(),
                ledger: self.ledger.clone(),
            },
        })
    }
}
struct Allocation {
    id: u64,
    raw: RawMemory,
    pool: Arc<dyn MemoryPool>,
    ledger: Arc<Ledger>,
}
// SAFETY: callers mutate only a uniquely owned MutableMemory. SharedMemory
// exposes immutable CPU views; Device permits imports only as read-only weights.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}
impl Drop for Allocation {
    fn drop(&mut self) {
        unsafe { self.pool.release(&self.raw) };
        self.ledger.owned.fetch_sub(self.raw.len, Ordering::AcqRel);
    }
}
pub struct MutableMemory {
    inner: Allocation,
}
impl MutableMemory {
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.inner.raw.ptr, self.inner.raw.len) }
    }
    pub fn seal(self) -> SharedMemory {
        SharedMemory {
            inner: Arc::new(self.inner),
        }
    }
}
#[derive(Clone)]
pub struct SharedMemory {
    inner: Arc<Allocation>,
}
impl fmt::Debug for SharedMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedMemory")
            .field("id", &self.id())
            .field("len", &self.len())
            .finish()
    }
}
impl SharedMemory {
    pub fn id(&self) -> u64 {
        self.inner.id
    }
    pub fn len(&self) -> usize {
        self.inner.raw.len
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.inner.raw.ptr, self.inner.raw.len) }
    }
    pub(crate) fn ptr(&self) -> *mut u8 {
        self.inner.raw.ptr
    }
    /// Fails while CPU readers or DSP imports retain another ownership reference.
    pub fn try_into_mut(self) -> Result<MutableMemory, Self> {
        match Arc::try_unwrap(self.inner) {
            Ok(inner) => Ok(MutableMemory { inner }),
            Err(inner) => Err(Self { inner }),
        }
    }
}
impl AsRef<[u8]> for SharedMemory {
    fn as_ref(&self) -> &[u8] { self.as_bytes() }
}
impl AsFd for SharedMemory {
    fn as_fd(&self) -> BorrowedFd<'_> {
        unsafe { BorrowedFd::borrow_raw(self.inner.raw.fd) }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::alloc::{alloc_zeroed, dealloc, Layout};
    use std::collections::HashMap;
    use std::fs::File;
    use std::os::fd::AsRawFd;
    use std::sync::Mutex;
    pub(crate) struct MockPool {
        allocations: Mutex<HashMap<usize, File>>,
        pub(crate) freed: AtomicUsize,
        fail: bool,
    }
    unsafe impl MemoryPool for MockPool {
        fn allocate(&self, n: usize) -> Result<RawMemory, String> {
            if self.fail {
                return Err("injected allocation failure".into());
            }
            let f = File::open("/dev/null").map_err(|e| e.to_string())?;
            let fd = f.as_raw_fd();
            let p = unsafe { alloc_zeroed(Layout::from_size_align(n, 4096).unwrap()) };
            if p.is_null() {
                return Err("mock allocation failed".into());
            }
            self.allocations.lock().unwrap().insert(p as usize, f);
            Ok(RawMemory {
                ptr: p,
                len: n,
                fd,
                token: 0,
            })
        }
        unsafe fn release(&self, r: &RawMemory) {
            assert!(self
                .allocations
                .lock()
                .unwrap()
                .remove(&(r.ptr as usize))
                .is_some());
            dealloc(r.ptr, Layout::from_size_align(r.len, 4096).unwrap());
            self.freed.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub(crate) fn pool(fail: bool) -> Arc<MockPool> {
        Arc::new(MockPool {
            allocations: Mutex::new(HashMap::new()),
            freed: AtomicUsize::new(0),
            fail,
        })
    }
    pub(crate) fn allocator(p: Arc<MockPool>, limit: usize) -> SharedAllocator {
        SharedAllocator::from_pool(p, limit).unwrap()
    }
    #[test]
    fn sealed_clone_retains_allocator_and_counts_once() {
        let p = pool(false);
        let a = SharedAllocator::from_pool(p.clone(), 8192).unwrap();
        let mut m = a.allocate(4096).unwrap();
        assert!(m.as_mut_bytes().iter().all(|&v| v == 0));
        m.as_mut_bytes()[13] = 55;
        let sealed = m.seal();
        let imported = sealed.clone();
        assert_eq!(sealed.id(), imported.id());
        assert_eq!(a.owned_bytes(), 4096);
        let sealed = sealed.try_into_mut().err().unwrap();
        drop(a);
        drop(sealed);
        assert_eq!(p.freed.load(Ordering::Relaxed), 0);
        assert_eq!(imported.as_bytes()[13], 55);
        drop(imported);
        assert_eq!(p.freed.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn mutation_requires_all_imports_and_readers_gone() {
        let p = pool(false);
        let a = SharedAllocator::from_pool(p.clone(), 8192).unwrap();
        let s = a.allocate(4096).unwrap().seal();
        let copy = s.clone();
        let s = s.try_into_mut().err().unwrap();
        drop(copy);
        let mut m = s.try_into_mut().ok().unwrap();
        m.as_mut_bytes()[0] = 77;
        let s = m.seal();
        assert_eq!(s.as_bytes()[0], 77);
        drop(s);
        assert_eq!(a.owned_bytes(), 0);
        assert_eq!(p.freed.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn concurrent_budget_is_reserved_before_allocator_call() {
        let p = pool(false);
        let a = SharedAllocator::from_pool(p.clone(), 4096).unwrap();
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8).map(|_| scope.spawn(|| a.allocate(4096))).collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
        assert_eq!(a.owned_bytes(), 4096);
        drop(results);
        assert_eq!(a.owned_bytes(), 0);
        assert_eq!(p.freed.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn bounds_budget_and_failed_alloc_do_not_leak_accounting() {
        let p = pool(false);
        let a = SharedAllocator::from_pool(p.clone(), 4096).unwrap();
        for n in [0, 1, 4095, (512 << 20) + 4096] {
            assert!(a.allocate(n).is_err());
        }
        let s = a.allocate(4096).unwrap().seal();
        assert!(a.allocate(4096).is_err());
        assert_eq!(a.owned_bytes(), 4096);
        drop(s);
        assert_eq!(a.owned_bytes(), 0);
        let failed = SharedAllocator::from_pool(pool(true), 4096).unwrap();
        assert!(failed.allocate(4096).is_err());
        assert_eq!(failed.owned_bytes(), 0);
    }
}
