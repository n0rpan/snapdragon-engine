//! UDMA-read-only control; explicitly NOT a matvec or a W8-K1 result.
//! Reuses the qualified GEMM pipeline with HMX/output writes disabled after
//! a real, numerically checked HMX control. Large T keeps the host asleep.
#[cfg(feature="device")]
fn run(go:&str)->Result<(),String>{
    use oxn::{Backend,GemmArgs,GF_YCOL,GF_NO_HMX,GF_NO_YWRITE};
    use oxn::tiles::{self,OutLayout};
    use std::time::{Instant,Duration,SystemTime,UNIX_EPOCH};
    use std::io::Write;
    let wall=||SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    if !go.is_empty(){
        println!("{{\"event\":\"ready_before_device\",\"pid\":{}}}",std::process::id());std::io::stdout().flush().map_err(|e|e.to_string())?;
        let until=Instant::now()+Duration::from_secs(30);
        while !std::path::Path::new(go).is_file(){
            if Instant::now()>until{return Err("guard handshake timeout; no device opened".into());}
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    let(t,k,o)=(1024usize,3072usize,9216usize);
    let mut be=oxn::device::Device::open(1)?;
    let x=be.alloc(0,t*k*2)?;let w=be.alloc(0,o*k*2)?;let y=be.alloc(0,t*o*2)?;
    // SAFETY: exclusive, live buffers with aligned CPU mappings; all values .125.
    unsafe{std::slice::from_raw_parts_mut(x.ptr.cast::<u16>(),x.len/2).fill(0x3000);
        std::slice::from_raw_parts_mut(w.ptr.cast::<u16>(),w.len/2).fill(0x3000);}
    let mut args=GemmArgs{x_slot:x.slot,w_slot:w.slot,y_slot:y.slot,t:32,k:k as u32,o:o as u32,flags:GF_YCOL,r:1,c:4,..Default::default()};
    be.gemm(0,&args)?;
    let lay=OutLayout{t_pad:32,o,nchunks:1,ycol:true};
    for row in [0,31]{let mut got=vec![0.0;o];
        // SAFETY: the completed control wrote exactly this tile layout in y.
        unsafe{tiles::read_out_row(y.ptr.cast(),&lay,row,0,&mut got);}
        if got.iter().any(|&v|v!=48.0){return Err("positive HMX numeric control failed".into());}
    }
    println!("{{\"event\":\"numeric_control\",\"t\":32,\"k\":{k},\"o\":{o},\"expected\":48.0,\"passed\":true}}");
    args.t=t as u32;args.flags=GF_YCOL|GF_NO_HMX|GF_NO_YWRITE;
    for _ in 0..3{be.gemm(0,&args)?;}
    let mut calls=0u64;let mut dsp_us=0u64;let mut steps=0u64;
    let t0=Instant::now();let begin=wall();
    println!("{{\"event\":\"timed_begin\",\"wall_ns\":{begin}}}");std::io::stdout().flush().map_err(|e|e.to_string())?;
    while t0.elapsed().as_secs_f64()<3.0{
        let st=be.gemm(0,&args)?;
        if st.r!=1||st.c!=4||st.steps!=2304{return Err(format!("unexpected DMA geometry r{} c{} steps{}",st.r,st.c,st.steps));}
        calls+=1;dsp_us+=st.dsp_us as u64;steps+=st.steps as u64;
    }
    let seconds=t0.elapsed().as_secs_f64();let end=wall();
    // One weight pass per token-row block; activations loaded once per block.
    let weight_bytes=calls*(o*k*2*(t/32))as u64;let activation_bytes=calls*(t*k*2)as u64;
    println!("{{\"event\":\"result\",\"kind\":\"DMA-read-only-NOT-matvec\",\"begin_wall_ns\":{begin},\"end_wall_ns\":{end},\"seconds\":{seconds:.9},\"calls\":{calls},\"steps\":{steps},\"dsp_us\":{dsp_us},\"weight_read_bytes\":{weight_bytes},\"activation_read_bytes\":{activation_bytes},\"output_write_bytes\":0,\"weight_GB_s\":{:.6},\"payload_read_GB_s\":{:.6}}}",weight_bytes as f64/seconds/1e9,(weight_bytes+activation_bytes)as f64/seconds/1e9);
    drop(be);Ok(())
}
fn main(){
    let a:Vec<String>=std::env::args().collect();
    println!("{{\"event\":\"plan\",\"kind\":\"DMA-read-control\",\"t\":1024,\"k\":3072,\"o\":9216,\"r\":1,\"c\":4,\"seconds\":3}}");
    if a.iter().any(|x|x=="--dry-run"){return;}
    #[cfg(feature="device")]{
        let go=a.windows(2).find(|p|p[0]=="--go-file").map(|p|p[1].as_str()).unwrap_or("");
        if let Err(e)=run(go){eprintln!("oxn-read-bandwidth: STOP {e}");std::process::exit(3);}
    }
    #[cfg(not(feature="device"))]{eprintln!("device feature required");std::process::exit(2);}
}
