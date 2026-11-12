//! Device backend (feature `device`): the oxn skel on the DSP, through the C host library
//! `npu/oxn/host/oxn_host.c`. Opening a device session is refused unless OXN_ATTENDED=1.

use crate::{
    shared_memory::{MemoryPool, RawMemory, SharedMemory},
    weight_imports::WeightImports,
    ImportedWeight,
};
use std::ffi::{c_char, c_int, c_void};
use std::os::fd::{AsFd, AsRawFd};
use std::sync::Mutex;

use crate::{Backend, Buf, GemmArgs, GemmStats};

#[repr(C)]
struct Opaque {
    _p: [u8; 0],
}
#[repr(C)]
struct MemoryPoolOpaque {
    _p: [u8; 0],
}
#[repr(C)]
struct MemoryOpaque {
    _p: [u8; 0],
}

/// Mirrors `struct oxn_rsp` in npu/oxn/common/oxn_proto.h (64 bytes). The host library checks
/// status and magic itself, so the Rust side only reads `v`.
#[repr(C)]
#[allow(dead_code)]
struct OxnRsp {
    status: i32,
    op: u32,
    v: [u32; 12],
    magic: u32,
    pad: u32,
}
const _: () = assert!(std::mem::size_of::<OxnRsp>() == 64);

const OP_GEMM: u32 = 6;
const GV_STEPS: usize = 5;
const GV_QTIMER_US: usize = 4;
const GV_ROWBLK: usize = 6;
const GV_COLBLK: usize = 7;
/// v[11] (OXN_V_DETAIL) on GEMM success: low 32 bits of the W8 expansion cycles
const GV_DEQ: usize = 11;
const GEMM_NARGS: usize = 13;
const ERR_LEN: usize = 512;

extern "C" {
    fn oxn_native_reference_sample(w:*const u8,nw:usize,x:*const u16,nx:usize,kind:u32,t:u32,k:u32,o:u32,experts:u32,e:u32,m:u32,n:u32,out:*mut f64)->c_int;
    fn oxn_host_open(
        out: *mut *mut Opaque,
        n_sessions: c_int,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn oxn_host_sessions(h: *const Opaque) -> c_int;
    fn oxn_host_vtcm_bytes(h: *const Opaque) -> u32;
    fn oxn_host_alloc_map(
        h: *mut Opaque,
        session: c_int,
        bytes: usize,
        slot: *mut u32,
        ptr: *mut *mut c_void,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn oxn_host_unmap_keep(
        h: *mut Opaque,
        session: c_int,
        slot: u32,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn oxn_host_remap(
        h: *mut Opaque,
        session: c_int,
        slot: u32,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn oxn_host_cmd(
        h: *mut Opaque,
        session: c_int,
        op: u32,
        a: *const u32,
        na: c_int,
        rsp: *mut OxnRsp,
        wall_s: *mut f64,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn oxn_host_failed(h: *const Opaque) -> c_int;
    fn oxn_host_close(h: *mut Opaque);
    fn oxn_host_window_bytes(h: *const Opaque, session: c_int) -> usize;
    fn oxn_host_set_window_limit(
        h: *mut Opaque,
        bytes: usize,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn oxn_host_window_admit(
        h: *const Opaque,
        session: c_int,
        bytes: usize,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn oxn_host_import_weight(
        h: *mut Opaque,
        session: c_int,
        fd: c_int,
        ptr: *mut c_void,
        bytes: usize,
        slot: *mut u32,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn oxn_host_release_weight(
        h: *mut Opaque,
        session: c_int,
        slot: u32,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn oxn_host_drain_weights(h: *mut Opaque, err: *mut c_char, errlen: usize) -> c_int;
    fn oxn_memory_pool_open(
        out: *mut *mut MemoryPoolOpaque,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn oxn_memory_pool_close(pool: *mut MemoryPoolOpaque);
    fn oxn_memory_alloc(
        pool: *mut MemoryPoolOpaque,
        bytes: usize,
        out: *mut *mut MemoryOpaque,
        ptr: *mut *mut c_void,
        fd: *mut c_int,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn oxn_memory_free(mem: *mut MemoryOpaque);
}

/// Pure host sample oracle; no DSP interaction. Bounds are checked again in C.
pub fn native_reference_sample(w:&[u8],x:&[u16],kind:u32,t:u32,k:u32,o:u32,experts:u32,e:u32,m:u32,n:u32)->Result<f64,String>{
    let mut value=0.;
    let rc=unsafe{oxn_native_reference_sample(w.as_ptr(),w.len(),x.as_ptr(),x.len(),kind,t,k,o,experts,e,m,n,&mut value)};
    if rc!=0||!value.is_finite(){Err("invalid native host reference sample".into())}else{Ok(value)}
}

#[cfg(test)]
mod native_reference_tests {
    use super::*;
    #[test]
    fn host_oracle_bounds_and_zero_weights_without_device() {
        let x=vec![0x3c00u16;32*512];
        for (kind,bytes) in [(3,880*4*2),(4,1152*4*2)] {
            let w=vec![0u8;bytes];
            assert_eq!(native_reference_sample(&w,&x,kind,32,512,32,1,0,31,31).unwrap(),0.);
            assert!(native_reference_sample(&w[..bytes-1],&x,kind,32,512,32,1,0,31,31).is_err());
            assert!(native_reference_sample(&w,&x[..x.len()-1],kind,32,512,32,1,0,31,31).is_err());
            assert!(native_reference_sample(&w,&x,kind,32,512,32,1,1,31,31).is_err());
            assert!(native_reference_sample(&w,&x,kind,32,u32::MAX,32,1,0,31,31).is_err());
        }
    }
}

pub(crate) struct CpuMemoryPool {
    h: Mutex<*mut MemoryPoolOpaque>,
}
// SAFETY: all allocator callbacks serialized by h; buffers retain this runtime.
unsafe impl Send for CpuMemoryPool {}
unsafe impl Sync for CpuMemoryPool {}
impl CpuMemoryPool {
    pub fn open() -> Result<Self, String> {
        let mut h = std::ptr::null_mut();
        let mut err = [0 as c_char; ERR_LEN];
        let rc = unsafe { oxn_memory_pool_open(&mut h, err.as_mut_ptr(), ERR_LEN) };
        if rc != 0 || h.is_null() {
            return Err(err_text(&err.map(|v| v as u8)));
        }
        Ok(Self { h: Mutex::new(h) })
    }
}
unsafe impl MemoryPool for CpuMemoryPool {
    fn allocate(&self, bytes: usize) -> Result<RawMemory, String> {
        let h = self.h.lock().map_err(|_| "CPU allocation mutex poisoned")?;
        let mut m = std::ptr::null_mut();
        let mut p = std::ptr::null_mut();
        let mut fd = -1;
        let mut err = [0 as c_char; ERR_LEN];
        let rc = unsafe {
            oxn_memory_alloc(
                *h,
                bytes,
                &mut m,
                &mut p,
                &mut fd,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        if rc != 0 {
            return Err(err_text(&err.map(|v| v as u8)));
        }
        Ok(RawMemory {
            ptr: p.cast(),
            len: bytes,
            fd,
            token: m as usize,
        })
    }
    unsafe fn release(&self, raw: &RawMemory) {
        let _h = self.h.lock().unwrap_or_else(|e| e.into_inner());
        oxn_memory_free(raw.token as *mut MemoryOpaque);
    }
}
impl Drop for CpuMemoryPool {
    fn drop(&mut self) {
        let h = self.h.get_mut().unwrap_or_else(|e| e.into_inner());
        unsafe { oxn_memory_pool_close(*h) };
        *h = std::ptr::null_mut();
    }
}

fn err_text(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// An open oxn device object: 1..8 FastRPC sessions on the DSP. One per process (the C library enforces it).
pub struct Device {
    h: *mut Opaque,
    imports: WeightImports,
    import_failed: bool,
}

// The session is only moved between threads, never used from two at once.
unsafe impl Send for Device {}

impl Device {
    /// Opens `n_sessions` (1..=8) FastRPC sessions. One session holds the HMX unit (resource mode HOLD);
    /// several take turns on it (resource mode PEROP).
    pub fn open(n_sessions: usize) -> Result<Device, String> {
        if !(1..=8).contains(&n_sessions) {
            return Err(format!(
                "device: {n_sessions} sessions requested, need 1..=8"
            ));
        }
        let mut h: *mut Opaque = std::ptr::null_mut();
        let mut err = [0 as c_char; ERR_LEN];
        let rc = unsafe { oxn_host_open(&mut h, n_sessions as c_int, err.as_mut_ptr(), ERR_LEN) };
        if rc != 0 || h.is_null() {
            let text = err_text(&err.map(|c| c as u8));
            return Err(format!("oxn_host_open rc {rc}: {text}"));
        }
        Ok(Device {
            h,
            imports: WeightImports::new(),
            import_failed: false,
        })
    }

    /// Unmap a buffer from this DSP session without freeing its CPU allocation.
    pub fn unmap_keep(&mut self, b: Buf) -> Result<(), String> {
        let mut err = [0 as c_char; ERR_LEN];
        let rc = unsafe {
            oxn_host_unmap_keep(
                self.h,
                b.session as c_int,
                b.slot,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(err_text(&err.map(|c| c as u8)))
        }
    }

    /// Re-import an existing allocation. Its CPU pointer stays unchanged; no copy.
    pub fn remap(&mut self, b: Buf) -> Result<(), String> {
        let mut err = [0 as c_char; ERR_LEN];
        let rc = unsafe {
            oxn_host_remap(
                self.h,
                b.session as c_int,
                b.slot,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(err_text(&err.map(|c| c as u8)))
        }
    }

    /// True once any call on this session has failed (the session refuses further device calls).
    pub fn failed(&self) -> bool {
        self.import_failed || self.h.is_null() || unsafe { oxn_host_failed(self.h) != 0 }
    }

    /// W2b per-PD VA budget, independent of the CPU allocation budget. Legacy
    /// default is unchanged. Admission errors make NO device call, not sticky.
    pub fn set_window_limit(&mut self, bytes: usize) -> Result<(), String> {
        let mut err = [0 as c_char; ERR_LEN];
        if unsafe { oxn_host_set_window_limit(self.h, bytes, err.as_mut_ptr(), ERR_LEN) } != 0 {
            return Err(err_text(&err.map(|v| v as u8)));
        }
        Ok(())
    }
    pub fn window_bytes(&self, session: usize) -> Result<usize, String> {
        if session >= self.sessions() {
            return Err("window session out of range".into());
        }
        let n = unsafe { oxn_host_window_bytes(self.h, session as c_int) };
        if n == usize::MAX {
            return Err("invalid window accounting".into());
        }
        Ok(n)
    }
    fn admit_window(&self, session: usize, bytes: usize) -> Result<(), String> {
        if session >= self.sessions() {
            return Err("window session out of range".into());
        }
        let mut err = [0 as c_char; ERR_LEN];
        if unsafe {
            oxn_host_window_admit(self.h, session as c_int, bytes, err.as_mut_ptr(), ERR_LEN)
        } != 0
        {
            return Err(err_text(&err.map(|v| v as u8)));
        }
        Ok(())
    }

    /// Bounded native expert pricing batch, all experts in one projection RPC.
    /// Inputs and outputs are expert-major HMX tiles (Y column-major); no model hook.
    pub fn native_batch(&mut self,x:Buf,w:Buf,y:Buf,t:u32,k:u32,o:u32,experts:u32,kind:u32)->Result<GemmStats,String>{
        if self.failed(){return Err("device failed or closed".into());}
        if x.session!=w.session||x.session!=y.session||x.session as usize>=self.sessions(){return Err("native batch session mismatch".into());}
        if self.imports.is_readonly(y.session,y.slot){return Err("native output is read-only weight slot".into());}
        let a=[x.slot,0,w.slot,0,y.slot,0,t,k,o,experts,kind,0,0,0];
        let mut rsp=OxnRsp{status:0,op:0,v:[0;12],magic:0,pad:0};let mut wall=0.;let mut err=[0 as c_char;ERR_LEN];
        let rc=unsafe{oxn_host_cmd(self.h,x.session as c_int,14,a.as_ptr(),14,&mut rsp,&mut wall,err.as_mut_ptr(),ERR_LEN)};
        if rc!=0{return Err(format!("native batch: {}",err_text(&err.map(|v|v as u8))));}
        let v=rsp.v;let q=|i:usize|v[i] as u64|((v[i+1] as u64)<<32);
        Ok(GemmStats{wall_s:wall,dsp_us:v[4],pcycles:q(0),wait_pcycles:q(2),hmx_pcycles:q(9),deq_pcycles:v[11] as u64,steps:v[5],r:v[6],c:v[7]})
    }

    /// Import sealed CPU weights without taking ownership of their allocation.
    pub fn import_weight(
        &mut self,
        session: usize,
        memory: &SharedMemory,
    ) -> Result<ImportedWeight, String> {
        if self.failed() {
            return Err("device failed or closed".into());
        }
        if session >= self.sessions() {
            return Err("import session out of range".into());
        }
        self.admit_window(session, memory.len())?;
        if self.imports.contains(session as u32, memory.id()) {
            return Err("weight already imported in this session".into());
        }
        self.imports.begin(memory); // retain BEFORE a potentially partial C import
        let mut slot = u32::MAX;
        let mut err = [0 as c_char; ERR_LEN];
        let rc = unsafe {
            oxn_host_import_weight(
                self.h,
                session as c_int,
                memory.as_fd().as_raw_fd(),
                memory.ptr().cast(),
                memory.len(),
                &mut slot,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        if rc != 0 {
            self.import_failed = true;
            return Err(err_text(&err.map(|v| v as u8)));
        }
        let h = self.imports.commit(session as u32, slot);
        if h.is_err() {
            self.import_failed = true;
        }
        h
    }
    pub fn release_weight(&mut self, handle: ImportedWeight) -> Result<(), String> {
        self.imports.validate(&handle)?;
        if self.failed() {
            return Err("device failed or closed".into());
        }
        let b = handle.buf();
        let mut err = [0 as c_char; ERR_LEN];
        let rc = unsafe {
            oxn_host_release_weight(
                self.h,
                b.session as c_int,
                b.slot,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        if rc != 0 {
            self.import_failed = true;
            return Err(err_text(&err.map(|v| v as u8)));
        }
        self.imports.release_confirmed(&handle)
    }
    /// Fallible shutdown for owners that must report release failure. Drop is a
    /// safety fallback; uncertain imports deliberately retain CPU storage.
    pub fn close(mut self) -> Result<(), String> {
        self.close_inner()
    }
    fn close_inner(&mut self) -> Result<(), String> {
        if self.h.is_null() {
            return Ok(());
        }
        let mut err = [0 as c_char; ERR_LEN];
        let mut release_error = None;
        if !self.imports.is_empty() {
            if self.failed() {
                release_error = Some("failed device: imported detach uncertain".to_owned());
            } else if unsafe { oxn_host_drain_weights(self.h, err.as_mut_ptr(), ERR_LEN) } != 0 {
                release_error = Some(err_text(&err.map(|v| v as u8)));
            }
        }
        unsafe { oxn_host_close(self.h) };
        self.h = std::ptr::null_mut();
        if let Some(e) = release_error {
            eprintln!("oxn: ERROR {e}; CPU imports retained until process exit");
            Err(e)
        } else {
            self.imports.clear_confirmed();
            Ok(())
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let _ = self.close_inner();
    }
}

impl Backend for Device {
    fn name(&self) -> &'static str {
        "device"
    }

    fn sessions(&self) -> usize {
        unsafe { oxn_host_sessions(self.h) as usize }
    }

    fn vtcm_bytes(&self) -> u32 {
        unsafe { oxn_host_vtcm_bytes(self.h) }
    }

    fn alloc(&mut self, session: usize, bytes: usize) -> Result<Buf, String> {
        if self.failed() {
            return Err("device failed or closed".into());
        }
        self.admit_window(session, bytes)?;
        let mut slot: u32 = 0;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let mut err = [0 as c_char; ERR_LEN];
        let rc = unsafe {
            oxn_host_alloc_map(
                self.h,
                session as c_int,
                bytes,
                &mut slot,
                &mut ptr,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        if rc != 0 {
            return Err(format!(
                "oxn_host_alloc_map: {}",
                err_text(&err.map(|c| c as u8))
            ));
        }
        Ok(Buf {
            session: session as u32,
            slot,
            ptr: ptr as *mut u8,
            len: bytes,
        })
    }

    fn gemm(&mut self, session: usize, g: &GemmArgs) -> Result<GemmStats, String> {
        if self.failed() {
            return Err("device failed or closed".into());
        }
        let mut a = [0u32; GEMM_NARGS];
        a[0] = g.x_slot;
        a[1] = g.x_off;
        a[2] = g.w_slot;
        a[3] = g.w_off;
        a[4] = g.y_slot;
        a[5] = g.y_off;
        a[6] = g.t;
        a[7] = g.k;
        a[8] = g.o;
        a[9] = 0; // VTCM cap: all acquired VTCM
        a[10] = g.flags;
        a[11] = g.r;
        a[12] = g.c;
        let mut rsp = OxnRsp {
            status: 0,
            op: 0,
            v: [0; 12],
            magic: 0,
            pad: 0,
        };
        let mut wall = 0f64;
        let mut err = [0 as c_char; ERR_LEN];
        let rc = unsafe {
            oxn_host_cmd(
                self.h,
                session as c_int,
                OP_GEMM,
                a.as_ptr(),
                GEMM_NARGS as c_int,
                &mut rsp,
                &mut wall,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        if rc != 0 {
            return Err(format!("oxn GEMM: {}", err_text(&err.map(|c| c as u8))));
        }
        let v = &rsp.v;
        let q = |lo: usize| v[lo] as u64 | ((v[lo + 1] as u64) << 32);
        Ok(GemmStats {
            wall_s: wall,
            dsp_us: v[GV_QTIMER_US],
            pcycles: q(0),
            wait_pcycles: q(2),
            hmx_pcycles: q(9),
            deq_pcycles: v[GV_DEQ] as u64,
            steps: v[GV_STEPS],
            r: v[GV_ROWBLK],
            c: v[GV_COLBLK],
        })
    }
}
