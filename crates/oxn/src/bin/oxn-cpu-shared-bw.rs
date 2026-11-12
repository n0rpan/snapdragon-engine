//! W2b prerequisite: identical CPU read kernel on anonymous vs shared DMA memory.
//! No model weights. Uses the existing skel only for a tiny positive GEMM control.
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::io::Write;

fn prefix(n:usize)->u64 {let r=(n%256)as u64;(n/256)as u64*32640+r*r.saturating_sub(1)/2}

/// Same eight-independent-accumulator NEON kernel as the lab memory roof probe.
/// The empty asm is a compiler memory clobber between passes, not a cache flush.
#[inline(never)]
unsafe fn read_chunk(p:*const u64,n:usize)->u64{
    #[cfg(target_arch="aarch64")]{
        use std::arch::aarch64::*;
        let mut a=[vdupq_n_u64(0);8];let mut i=0;
        while n-i>=16{
            for j in 0..8{a[j]=vaddq_u64(a[j],vld1q_u64(p.add(i+2*j)));}
            i+=16;
        }
        let mut sum=0u64;for v in a{sum=sum.wrapping_add(vaddvq_u64(v));}
        for j in i..n{sum=sum.wrapping_add(*p.add(j));}sum
    }
    #[cfg(not(target_arch="aarch64"))]{(0..n).fold(0u64,|s,i|s.wrapping_add(*p.add(i)))}
}

fn read(pool:&sched::Pool,bases:&[usize],words_per_buffer:usize,passes:usize)->Result<(f64,u64),String>{
    use sched::Team;
    let total=words_per_buffer.checked_mul(bases.len()).ok_or("size overflow")?;
    let checks:Vec<_>=(0..pool.nthreads()).map(|_|sched::Padded(AtomicU64::new(0))).collect();
    let start=Instant::now();
    pool.run_job(&|ctx:&mut sched::Ctx|{
        let begin=total*ctx.tid/pool.nthreads();let end=total*(ctx.tid+1)/pool.nthreads();let mut sum=0u64;
        for _ in 0..passes{
            // SAFETY: no instructions, default asm memory effects prevent hoisting
            // immutable loads across passes. All buffers stay live/read-only.
            unsafe{std::arch::asm!("",options(nostack,preserves_flags));}
            for (i,&base)in bases.iter().enumerate(){
                let b=begin.max(i*words_per_buffer);let e=end.min((i+1)*words_per_buffer);
                if b<e{sum=sum.wrapping_add(unsafe{read_chunk((base as *const u64).add(b-i*words_per_buffer),e-b)});}
            }
        }
        checks[ctx.tid].store(sum,Ordering::Relaxed);
    });
    let seconds=start.elapsed().as_secs_f64();let mut sum=0u64;
    for(t,c)in checks.iter().enumerate(){
        let expected=(prefix(total*(t+1)/pool.nthreads())-prefix(total*t/pool.nthreads()))*passes as u64;
        let got=c.load(Ordering::Relaxed);if got!=expected{return Err(format!("CPU{t} checksum {got} != {expected}"));}sum+=got;
    }
    Ok((seconds,sum))
}

#[cfg(feature="device")]
fn run(go:&str)->Result<(),String>{
    use oxn::{Backend,GemmArgs,GF_YCOL,tiles::{self,OutLayout}};
    if go.is_empty(){return Err("mandatory --go-file guard handshake missing".into());}
    println!("{{\"event\":\"ready_before_device\",\"pid\":{}}}",std::process::id());std::io::stdout().flush().map_err(|e|e.to_string())?;
    let deadline=Instant::now()+Duration::from_secs(30);
    while !std::path::Path::new(go).is_file(){if Instant::now()>deadline{return Err("guard handshake timeout, device unopened".into());}std::thread::sleep(Duration::from_millis(20));}
    let mut be=oxn::device::Device::open(1)?;
    let a=be.alloc(0,512<<20)?;let b=be.alloc(0,512<<20)?;
    let x=be.alloc(0,4096)?;let w=be.alloc(0,4096)?;let y=be.alloc(0,4096)?;
    // SAFETY: exclusive live allocations, GEMM input exactly32x32 halfs.
    unsafe{std::slice::from_raw_parts_mut(x.ptr.cast::<u16>(),1024).fill(0x3000);
        std::slice::from_raw_parts_mut(w.ptr.cast::<u16>(),1024).fill(0x3000);}
    be.gemm(0,&GemmArgs{x_slot:x.slot,w_slot:w.slot,y_slot:y.slot,t:32,k:32,o:32,flags:GF_YCOL,..Default::default()})?;
    let mut got=[0f32;32];
    unsafe{tiles::read_out_row(y.ptr.cast(),&OutLayout{t_pad:32,o:32,nchunks:1,ycol:true},0,0,&mut got);}
    if got.iter().any(|&v|v!=0.5){return Err("positive device numeric control failed".into());}
    let words=(1usize<<30)/8;let wb=words/2;
    let mut anon=vec![0u64;words];
    for(i,v)in anon.iter_mut().enumerate(){*v=(i&255)as u64;}
    for(i,buf)in [a,b].iter().enumerate(){
        // SAFETY: exclusive CPU ownership, no DSP read in the CPU benchmark.
        let target=unsafe{std::slice::from_raw_parts_mut(buf.ptr.cast::<u64>(),wb)};
        for(j,v)in target.iter_mut().enumerate(){*v=((i*wb+j)&255)as u64;}
    }
    let cpus:Vec<usize>=(0..8).collect();let pool=sched::Pool::new(&cpus,Duration::from_micros(100)).map_err(|e|e.to_string())?;
    let anon_base=[anon.as_ptr()as usize];let dma_base=[a.ptr as usize,b.ptr as usize];
    println!("{{\"event\":\"packed\",\"positive_gemm\":true,\"dma_reserved_bytes\":{},\"anonymous_bytes\":{},\"cpus\":[0,1,2,3,4,5,6,7],\"passes\":256}}",(1usize<<30)+3*4096,1usize<<30);std::io::stdout().flush().map_err(|e|e.to_string())?;
    for (i,shared)in [false,true,true,false].into_iter().enumerate(){
        let(bases,wb)=if shared{(&dma_base[..],wb)}else{(&anon_base[..],words)};
        read(&pool,bases,wb,4)?;
        let(seconds,checksum)=read(&pool,bases,wb,256)?;let gb_s=(1u64<<30)as f64*256.0/seconds/1e9;
        println!("{{\"event\":\"cpu_read_result\",\"index\":{i},\"kind\":\"{}\",\"seconds\":{seconds:.9},\"GB_s\":{gb_s:.6},\"checksum\":{checksum},\"bytes\":{}}}",if shared{"shared-dma"}else{"anonymous"},(1u64<<30)*256);std::io::stdout().flush().map_err(|e|e.to_string())?;
    }
    drop(pool);drop(be);drop(anon);
    println!("{{\"event\":\"complete\",\"passed\":true}}");Ok(())
}
fn main(){
    let a:Vec<String>=std::env::args().collect();
    if a.iter().any(|s|s=="--dry-run"){println!("{{\"kind\":\"CPU anonymous-vs-shared-DMA ABBA\",\"dma_bytes\":1073754112,\"host_bytes\":1073741824,\"threads\":8,\"array_bytes\":1073741824,\"model\":false,\"new_skel\":false}}");return;}
    #[cfg(feature="device")]{let go=a.windows(2).find(|p|p[0]=="--go-file").map_or("",|p|p[1].as_str());if let Err(e)=run(go){eprintln!("oxn-cpu-shared-bw: STOP {e}");std::process::exit(3);}}
    #[cfg(not(feature="device"))]{eprintln!("device feature required");std::process::exit(2);}
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn checksum_kernel_and_non_aligned_partitions(){
        let data:Vec<u64>=(0..1025).map(|i|(i&255)as u64).collect();
        for n in 0..=1025{assert_eq!(unsafe{read_chunk(data.as_ptr(),n)},prefix(n));}
        for n in [17,255,1025]{for t in 0..7{let b=n*t/7;let e=n*(t+1)/7;assert_eq!(unsafe{read_chunk(data.as_ptr().add(b),e-b)},prefix(e)-prefix(b));}}
    }
}
