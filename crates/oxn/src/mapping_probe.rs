//! Synthetic checks shared by mapping probes. These raw buffers are valid only
//! while owned by their backend; callers must uphold that ownership contract.
use crate::{Backend, Buf, GemmArgs, GF_YCOL};
use crate::tiles::{self, OutLayout};

/// Write a diagonal matrix to one tile of a live, exclusively owned CPU buffer.
/// # Safety
/// `w.ptr` must address `w.len` writable bytes owned by the live backend; no DSP
/// or other thread may access this tile until the write completes.
pub unsafe fn write_diagonal(w: Buf, off: usize, diagonal: f32) -> Result<(),String> {
    if off.checked_add(2048).is_none_or(|end|end>w.len) {return Err("weight tile bounds".into());}
    if !diagonal.is_finite() || diagonal.abs()>65504.0 {return Err("invalid diagonal".into());}
    let tile=unsafe{std::slice::from_raw_parts_mut(w.ptr.add(off),2048)};
    tile.fill(0);
    for i in 0..32 {
        let p=tiles::wt_elem_idx(i,i)*2;
        tile[p..p+2].copy_from_slice(&tiles::f32_to_half(diagonal).to_le_bytes());
    }
    Ok(())
}

/// Check X(identity)×W(diagonal), without rewriting W. This can detect whether
/// CPU changes made while W was unmapped reached a fresh DSP mapping.
/// # Safety
/// All buffers must be live allocations of `be`/`session`, with X already holding
/// the identity and W the expected diagonal. Y is at least4096 writable bytes;
/// no concurrent caller may access these buffers or run this backend.
pub unsafe fn verify_diagonal(be:&mut dyn Backend,session:usize,x:Buf,y:Buf,w:Buf,off:usize,expected:f32)->Result<(),String>{
    if off.checked_add(2048).is_none_or(|end|end>w.len) || x.len<4096 || y.len<4096 {
        return Err("mapping check bounds".into());
    }
    let off32=u32::try_from(off).map_err(|_| "mapping offset exceeds 32-bit protocol")?;
    be.gemm(session,&GemmArgs{x_slot:x.slot,w_slot:w.slot,w_off:off32,
        y_slot:y.slot,t:32,k:32,o:32,flags:GF_YCOL,..Default::default()})?;
    let lay=OutLayout{t_pad:32,o:32,nchunks:1,ycol:true};
    for row in 0..32 {
        let mut got=[0.0;32];
        unsafe{tiles::read_out_row(y.ptr.cast(),&lay,row,0,&mut got);}
        for(col,value)in got.iter().enumerate(){
            if *value != if row==col {expected}else{0.0}{
                return Err(format!("mapping check session{session} slot{} offset{off} row{row} col{col}: {value}",w.slot));
            }
        }
    }
    Ok(())
}

/// Fill and check one identity tile.
/// # Safety
/// The ownership, allocation and exclusion contracts of both helpers apply.
pub unsafe fn check_identity(be:&mut dyn Backend,session:usize,x:Buf,y:Buf,w:Buf,off:usize)->Result<(),String>{
    unsafe{write_diagonal(w,off,1.0)?;verify_diagonal(be,session,x,y,w,off,1.0)}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiles::ActLayout;
    #[test]
    fn checks_cpu_changes_and_rejects_stale_diagonal() {
        let mut be=crate::emu::Emu::new(8<<20,1);
        let x=be.alloc(0,4096).unwrap();let y=be.alloc(0,4096).unwrap();let w=be.alloc(0,8192).unwrap();
        for row in 0..32 {
            let mut a=[0.0;32];a[row]=1.0;
            // SAFETY: all three buffers are live/exclusive and sufficiently large.
            unsafe { tiles::write_act_row(x.ptr.cast(),&ActLayout::new(32,32,32),row,0,&a); }
        }
        unsafe {
            check_identity(&mut be,0,x,y,w,0).unwrap();
            write_diagonal(w,6144,2.0).unwrap();
            verify_diagonal(&mut be,0,x,y,w,6144,2.0).unwrap();
            assert!(verify_diagonal(&mut be,0,x,y,w,6144,1.0).is_err());
            assert!(write_diagonal(w,8192,1.0).is_err());
        }
    }
}
