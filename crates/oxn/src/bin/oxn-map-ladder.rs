//! Mapping reachability proof, not an arithmetic-throughput benchmark.
//! Synthetic weights only; eight sessions maximum, 128MiB chunks, stop on first
//! refusal. Every chunk's first and last tile must be readable by the DSP.
use std::time::Instant;
use oxn::Backend;
use oxn::mapping_probe::check_identity;
use oxn::tiles::{self, ActLayout};
const MIB: usize = 1 << 20;
const CHUNK: usize = 128 * MIB;

fn run(mut be: Box<dyn Backend>, chunks: usize, minimum_gib: usize, target: usize) -> Result<(), String> {
    let ns=be.sessions(); let start=Instant::now();
    let mut inputs=Vec::new(); let mut outputs=Vec::new(); let mut held=Vec::new();
    for s in 0..ns {
        let x=be.alloc(s,4096)?; let y=be.alloc(s,4096)?;
        let lay=ActLayout::new(32,32,32);
        for row in 0..32 {
            let mut src=[0.0;32]; src[row]=1.0;
            // SAFETY: x is exclusive, 4096 bytes; this layout writes exactly2048.
            unsafe { tiles::write_act_row(x.ptr.cast(), &lay,row,0,&src); }
        }
        inputs.push(x);outputs.push(y);
    }
    let mut mapped=ns*8192;
    for round in 0..chunks {
        for s in 0..ns {
            if mapped >= target { break; }
            let avail=std::fs::read_to_string("/proc/meminfo").map_err(|e|e.to_string())?
                .lines().find_map(|l| l.strip_prefix("MemAvailable:").and_then(|v|v.split_whitespace().next()).and_then(|v|v.parse::<usize>().ok()))
                .ok_or("MemAvailable unavailable")? *1024;
            if avail < minimum_gib*(1<<30)+CHUNK { return Err(format!("host admission stop available={avail} mapped={mapped}")); }
            let t=Instant::now(); let w=be.alloc(s,CHUNK)?;
            // Touch each page on the CPU as an explicit residency/coherence proof,
            // not just successful reservation of virtual addresses.
            unsafe { for off in (0..w.len).step_by(4096) { w.ptr.add(off).write_volatile(0); } }
            // SAFETY: these buffers belong to this backend, used exclusively in this loop.
            unsafe { check_identity(&mut *be,s,inputs[s],outputs[s],w,0)?; }
            unsafe { check_identity(&mut *be,s,inputs[s],outputs[s],w,CHUNK-2048)?; }
            mapped+=w.len;held.push(w);
            println!("{{\"event\":\"mapped\",\"session\":{s},\"round\":{round},\"session_bytes\":{},\"total_bytes\":{mapped},\"available_bytes\":{avail},\"seconds\":{:.6},\"first_last_tile\":true}}",(round+1)*CHUNK+8192,t.elapsed().as_secs_f64());
            use std::io::Write; std::io::stdout().flush().map_err(|e|e.to_string())?;
        }
    }
    // Re-check early mappings after every session has reached its highest total.
    for (i,&w) in held.iter().enumerate() {
        let s=i%ns;
        // SAFETY: all previously allocated buffers are still owned by this backend.
        unsafe { check_identity(&mut *be,s,inputs[s],outputs[s],w,CHUNK-2048)?; }
    }
    println!("{{\"event\":\"complete\",\"mapped_bytes\":{mapped},\"sessions\":{ns},\"max_chunks_per_session\":{chunks},\"all_mappings_rechecked\":true,\"seconds\":{:.6}}}",start.elapsed().as_secs_f64());
    // Backend owns every allocation; its Drop unmaps in reverse order.
    drop(be);Ok(())
}
fn main() {
    let args:Vec<String>=std::env::args().collect();
    let get=|key:&str,default:&str|args.windows(2).find(|p|p[0]==key).map(|p|p[1].as_str()).unwrap_or(default).to_owned();
    let backend=get("--backend","emu");
    let number=|key,default|get(key,default).parse::<usize>().unwrap_or_else(|_|{eprintln!("bad {key}");std::process::exit(2)});
    let ns=number("--sessions","4");let chunks=number("--chunks","30");let minimum=number("--host-floor-gib","6");
    if !(1..=8).contains(&ns)||!(1..=30).contains(&chunks)||minimum<4||minimum>16 {eprintln!("unsafe options");std::process::exit(2)}
    let total=ns*(8192+chunks*CHUNK);
    let target_mib=number("--target-mib","0");
    let target=if target_mib==0 {total} else {match target_mib.checked_mul(MIB) {Some(v) if v<=total=>v,_=>{eprintln!("target exceeds slot capacity");std::process::exit(2)}}};
    println!("{{\"event\":\"plan\",\"backend\":\"{backend}\",\"sessions\":{ns},\"chunks\":{chunks},\"slot_capacity_bytes\":{total},\"target_bytes\":{target},\"host_floor_gib\":{minimum}}}");
    if args.iter().any(|a|a=="--dry-run"){return;}
    let go=get("--go-file", "");
    if !go.is_empty() {
        println!("{{\"event\":\"ready_before_device\",\"pid\":{}}}",std::process::id());
        use std::io::Write;std::io::stdout().flush().unwrap();
        let until=Instant::now()+std::time::Duration::from_secs(30);
        while !std::path::Path::new(&go).is_file() {
            if Instant::now()>until { eprintln!("guard handshake timeout: no device opened");std::process::exit(2); }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
    let be:Result<Box<dyn Backend>,String>=match backend.as_str(){
        "emu"=>Ok(Box::new(oxn::emu::Emu::new(8<<20,ns))),
        #[cfg(feature="device")]
        "device"=>oxn::device::Device::open(ns).map(|b|Box::new(b) as Box<dyn Backend>),
        _=>Err("unknown/uncompiled backend".into()),
    };
    let result=be.and_then(|b|run(b,chunks,minimum,target));
    if let Err(e)=result {eprintln!("oxn-map-ladder: STOP {e}");std::process::exit(3)}
}
