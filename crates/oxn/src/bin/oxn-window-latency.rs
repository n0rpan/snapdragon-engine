//! Already-allocated whole-buffer remap timing. This does not yet import an
//! offset slice from a larger fd. CPU allocations and pointers stay unchanged.
#[cfg(feature="device")]
use std::time::Instant;
const SIZES: &[usize] = &[16,64,128,256,512];

#[cfg(feature="device")]
fn run(go: &str, settle_ms:u64) -> Result<(),String> {
    use oxn::{Backend,tiles::{self,ActLayout},mapping_probe::{check_identity,write_diagonal,verify_diagonal}};
    use oxn::device::Device;
    if !go.is_empty() {
        println!("{{\"event\":\"ready_before_device\",\"pid\":{}}}",std::process::id());
        use std::io::Write;std::io::stdout().flush().map_err(|e|e.to_string())?;
        let until=Instant::now()+std::time::Duration::from_secs(30);
        while !std::path::Path::new(go).is_file() {
            if Instant::now()>until {return Err("guard handshake timeout; no device opened".into());}
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
    let mut be=Device::open(1)?;
    let x=be.alloc(0,4096)?;let y=be.alloc(0,4096)?;
    for row in 0..32 {
        let mut a=[0.0;32];a[row]=1.0;
        // SAFETY: exclusive4096B x contains the2048B tile.
        unsafe{tiles::write_act_row(x.ptr.cast(),&ActLayout::new(32,32,32),row,0,&a);}
    }
    for &mib in SIZES {
        let w=be.alloc(0,mib<<20)?;
        let ptr=w.ptr;
        unsafe { check_identity(&mut be,0,x,y,w,0)?; }
        unsafe { check_identity(&mut be,0,x,y,w,w.len-2048)?; }
        // Drop the first map cycle. Numeric/coherence checks are outside timers.
        for rep in 0..11 {
            let t=Instant::now();be.unmap_keep(w)?;let unmap=t.elapsed().as_secs_f64();
            // Explicit diagnostic, outside the map/unmap timers. A required
            // delay still counts toward any eventual end-to-end policy cost.
            if settle_ms>0 {std::thread::sleep(std::time::Duration::from_millis(settle_ms));}
            // Host pointer remains usable while DSP has no mapping. Modify both
            // matrices on the CPU; verify after import without another CPU rewrite.
            let diagonal=if rep%2==0 {2.0}else{1.0};
            // SAFETY: no DSP mapping or concurrent user exists; CPU allocation lives.
            unsafe{write_diagonal(w,0,diagonal)?;write_diagonal(w,w.len-2048,diagonal)?;}
            let t=Instant::now();be.remap(w)?;let map=t.elapsed().as_secs_f64();
            if w.ptr!=ptr {return Err("CPU allocation pointer changed".into());}
            // SAFETY: imported buffers are live/exclusive; W is not rewritten here.
            unsafe{verify_diagonal(&mut be,0,x,y,w,0,diagonal)?;
                verify_diagonal(&mut be,0,x,y,w,w.len-2048,diagonal)?;}
            println!("{{\"event\":\"remap\",\"mib\":{mib},\"rep\":{rep},\"warmup\":{},\"map_ms\":{:.6},\"unmap_ms\":{:.6},\"settle_ms\":{settle_ms},\"pair_plus_settle_ms\":{:.6},\"pointer_stable\":true,\"first_last_tile\":true}}",rep==0,map*1e3,unmap*1e3,(map+unmap)*1e3+settle_ms as f64);
            use std::io::Write;std::io::stdout().flush().map_err(|e|e.to_string())?;
        }
        // Keep CPU allocation, release its DSP VA before the next size.
        be.unmap_keep(w)?;
    }
    drop(be);Ok(())
}
fn main(){
    let args:Vec<String>=std::env::args().collect();
    let settle_ms=match args.windows(2).find(|p|p[0]=="--settle-ms") {
        Some(p)=>match p[1].parse::<u64>(){Ok(n)if n<=100=>n,_=>{eprintln!("settle must be0..100ms");std::process::exit(2);}},
        None=>0
    };
    println!("{{\"event\":\"plan\",\"sizes_mib\":{:?},\"reps\":10,\"drop_first\":true,\"kind\":\"whole-buffer-import\"}}",SIZES);
    if args.iter().any(|a|a=="--dry-run"){return;}
    #[cfg(feature="device")]
    {
        let go=args.windows(2).find(|p|p[0]=="--go-file").map(|p|p[1].as_str()).unwrap_or("");
        if let Err(e)=run(go,settle_ms){eprintln!("oxn-window-latency: STOP {e}");std::process::exit(3);}
    }
    #[cfg(not(feature="device"))]
    {eprintln!("device feature required");std::process::exit(2);}
}
