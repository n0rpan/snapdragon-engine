//! Ownership bookkeeping separate from the C/DSP table, with stale-handle checks.
use crate::{shared_memory::SharedMemory, Buf};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT_DEVICE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct ImportedWeight {
    buf: Buf,
    device: u64,
    generation: u64,
}
impl ImportedWeight {
    /// Read-only operand descriptor. Using this as an output is refused by host
    /// validation. It is valid only until explicit release or backend teardown.
    pub fn buf(&self) -> Buf {
        self.buf
    }
}
struct Held {
    memory: SharedMemory,
    generation: u64,
}
pub(crate) struct WeightImports {
    device: u64,
    next: u64,
    pending: Vec<SharedMemory>,
    held: HashMap<(u32, u32), Held>,
}
impl WeightImports {
    pub fn new() -> Self {
        Self {
            device: NEXT_DEVICE
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
                .expect("device identity exhausted"),
            next: 1,
            pending: Vec::new(),
            held: HashMap::new(),
        }
    }
    pub fn is_readonly(&self, session:u32,slot:u32)->bool { self.held.contains_key(&(session,slot)) }
    pub fn contains(&self, session: u32, allocation: u64) -> bool {
        self.held
            .iter()
            .any(|((s, _), v)| *s == session && v.memory.id() == allocation)
    }
    /// Clone BEFORE invoking C; even an unsuccessful import can leave a driver mapping.
    pub fn begin(&mut self, memory: &SharedMemory) {
        self.pending.push(memory.clone());
    }
    pub fn commit(&mut self, session: u32, slot: u32) -> Result<ImportedWeight, String> {
        if self.held.contains_key(&(session, slot)) || self.pending.is_empty() {
            return Err("import ownership table mismatch".into());
        }
        let generation = self.next;
        self.next = self
            .next
            .checked_add(1)
            .ok_or("import generation exhausted")?;
        let memory = self.pending.pop().unwrap();
        let buf = Buf {
            session,
            slot,
            ptr: memory.ptr(),
            len: memory.len(),
        };
        self.held
            .insert((session, slot), Held { memory, generation });
        Ok(ImportedWeight {
            buf,
            device: self.device,
            generation,
        })
    }
    pub fn validate(&self, h: &ImportedWeight) -> Result<(), String> {
        if h.device != self.device
            || self
                .held
                .get(&(h.buf.session, h.buf.slot))
                .is_none_or(|v| v.generation != h.generation)
        {
            return Err("stale or foreign imported weight handle".into());
        }
        Ok(())
    }
    /// Only after confirmed DSP+driver detach.
    pub fn release_confirmed(&mut self, h: &ImportedWeight) -> Result<(), String> {
        self.validate(h)?;
        self.held.remove(&(h.buf.session, h.buf.slot));
        Ok(())
    }
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty() && self.held.is_empty()
    }
    /// Only after confirmed detach of EVERY imported/pending buffer.
    pub fn clear_confirmed(&mut self) {
        self.held.clear();
        self.pending.clear();
    }
    fn unique_bytes(&self) -> usize {
        let mut seen = HashSet::new();
        self.pending
            .iter()
            .chain(self.held.values().map(|v| &v.memory))
            .filter(|m| seen.insert(m.id()))
            .map(SharedMemory::len)
            .sum()
    }
}
impl Drop for WeightImports {
    fn drop(&mut self) {
        if self.is_empty() {
            return;
        }
        eprintln!("oxn: ERROR imported detach uncertain; retaining {} unique CPU bytes until process exit",self.unique_bytes());
        // Safety over reclaim: a failed/unknown unmap must never release the last
        // CPU allocation reference while the DSP might still reach it.
        for (_, held) in self.held.drain() {
            std::mem::forget(held.memory);
        }
        for memory in self.pending.drain(..) {
            std::mem::forget(memory);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared_memory::tests::{allocator, pool};
    #[test]
    fn import_keeps_cpu_alive_and_slot_reuse_rejects_stale_handle() {
        let p = pool(false);
        let a = allocator(p.clone(), 8192);
        let cpu = a.allocate(4096).unwrap().seal();
        let mut d = WeightImports::new();
        d.begin(&cpu);
        let h = d.commit(0, 3).unwrap();
        assert!(d.contains(0, cpu.id()));
        assert!(!d.contains(1, cpu.id()));
        let stale = ImportedWeight {
            buf: h.buf,
            device: h.device,
            generation: h.generation,
        };
        assert_eq!(a.owned_bytes(), 4096);
        assert!(cpu.clone().try_into_mut().is_err());
        d.release_confirmed(&h).unwrap();
        assert!(!d.contains(0, cpu.id()));
        d.begin(&cpu);
        let next = d.commit(0, 3).unwrap();
        assert!(d.validate(&stale).is_err());
        let foreign = WeightImports::new();
        assert!(foreign.validate(&next).is_err());
        drop(cpu);
        assert_eq!(p.freed.load(Ordering::Relaxed), 0);
        d.release_confirmed(&next).unwrap();
        assert_eq!(a.owned_bytes(), 0);
        assert_eq!(p.freed.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn pending_import_retains_until_confirmed_teardown() {
        let p = pool(false);
        let a = allocator(p.clone(), 4096);
        let cpu = a.allocate(4096).unwrap().seal();
        let mut d = WeightImports::new();
        d.begin(&cpu);
        drop(cpu);
        assert_eq!(a.owned_bytes(), 4096);
        assert_eq!(p.freed.load(Ordering::Relaxed), 0);
        d.clear_confirmed();
        assert_eq!(a.owned_bytes(), 0);
        assert_eq!(p.freed.load(Ordering::Relaxed), 1);
    }
}
