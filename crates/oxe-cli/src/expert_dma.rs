//! One real packed expert tensor in a temporary shared copy. No Engine inference,
//! no new weight layout, and no claim of whole-model decode throughput.
use super::*;
use oxn::Backend;
use std::sync::Mutex;
use sched::Team;

pub(super) fn run(a:&Args)->Result<(),String>{
    let s=setup(a,512);
    let name=a.get("tensor").ok_or("--tensor NAME required")?;
    let id=s.loaded.tensors.iter().position(|t|t.name==name).ok_or("unknown tensor")?;
    let w=*s.loaded.ws.get(id);
    if w.is_f32()||w.n_expert<8||w.bytes==0||w.bytes>512<<20{return Err("bounded quantized expert tensor required".into());}
    let q=w.qtype();let steps=a.num("runs",16384);
    if steps<256||steps>16384{return Err("runs256..16384 required".into());}
    let go=a.get("npu-go-file").ok_or("guard --npu-go-file required")?;
    println!("{}",json!({"event":"ready_before_device","pid":std::process::id()}));std::io::stdout().flush().map_err(|e|e.to_string())?;
    let deadline=Instant::now()+Duration::from_secs(30);
    while !Path::new(go).is_file(){if Instant::now()>deadline{return Err("guard handshake timeout, device unopened".into());}std::thread::sleep(Duration::from_millis(20));}
    let mut backend=oxn::device::Device::open(1)?;
    let dma=backend.alloc(0,w.bytes.div_ceil(4096)*4096)?;
    let source=unsafe{std::slice::from_raw_parts(w.ptr as *const u8,w.bytes)};
    let target=unsafe{std::slice::from_raw_parts_mut(dma.ptr,w.bytes)};target.copy_from_slice(source);
    if source!=target{return Err("packed weight copy differs".into());}
    let copied_hash=format!("{:016x}",model::weights::hash_bytes(target));
    let x:Vec<f32>=(0..w.k).map(|i|((i*17%257)as f32-128.)/127.).collect();
    let mut act=vec![0u8;q.act().bytes(w.k)];kernels_sys::quantize_act(q.act(),&x,&mut act);
    let outputs:Vec<_>=(0..s.pool.nthreads()).map(|_|Mutex::new(vec![0f32;w.rows])).collect();
    let run=|base:usize,steps:usize|->(f64,Vec<Vec<u32>>){
        let start=Instant::now();
        s.pool.run_job(&|ctx:&mut sched::Ctx|{
            let mut out=outputs[ctx.tid].lock().unwrap();
            for i in 0..steps{
                let expert=(i*73+ctx.tid*29)%w.n_expert;
                // SAFETY: source and temporary DMA copy are the SAME validated
                // packed layout; each worker owns its output and reads one expert.
                let weights=unsafe{std::slice::from_raw_parts((base+expert*w.expert_bytes())as *const u8,w.expert_bytes())};
                kernels_sys::gemv(q,w.k,&mut out,weights,&act);std::hint::black_box(&*out);
            }
        });
        let seconds=start.elapsed().as_secs_f64();
        let bits=outputs.iter().map(|v|v.lock().unwrap().iter().map(|v|v.to_bits()).collect()).collect();
        (seconds,bits)
    };
    println!("{}",json!({"event":"packed","model_sha256":s.loaded.ws.info.model_sha256,"tensor":name,"quant":q.name(),"shape":[w.k,w.rows,w.n_expert],"copied_bytes":w.bytes,"dma_reserved_bytes":dma.len,"copied_hash":copied_hash,"cpus":s.pool.cpus(),"steps":steps}));std::io::stdout().flush().map_err(|e|e.to_string())?;
    let mut expected=None;let mut rows=Vec::new();
    for (i,shared)in[false,true,true,false].into_iter().enumerate(){
        let ptr=if shared{dma.ptr as usize}else{w.ptr};run(ptr,steps);
        let(seconds,bits)=run(ptr,steps);
        if let Some(ref reference)=expected{if *reference!=bits{return Err("CPU packed GEMV differs from file-backed weights".into());}}else{expected=Some(bits);}
        let bytes=steps as u64*s.pool.nthreads()as u64*w.expert_bytes()as u64;
        let row=json!({"event":"expert_mv_result","index":i,"kind":if shared{"shared-dma"}else{"file-backed"},"seconds":seconds,"logical_weight_bytes":bytes,"logical_GB_s":bytes as f64/seconds/1e9,"output_bits_identical":true});
        println!("{row}");std::io::stdout().flush().map_err(|e|e.to_string())?;rows.push(row);
    }
    drop(backend);
    let result=json!({"passed":true,"kind":"W2b packed-expert CPU GEMV backing comparison, not whole-model decode","model_sha256":s.loaded.ws.info.model_sha256,"tensor":name,"cpus":s.pool.cpus(),"bytes":w.bytes,"dma_reserved_bytes":dma.len,"rows":rows});
    if let Some(path)=a.get("out"){std::fs::write(path,serde_json::to_vec_pretty(&result).unwrap()).map_err(|e|e.to_string())?;}
    println!("{}",json!({"event":"complete","passed":true}));Ok(())
}
