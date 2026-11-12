//! Original-GGUF -> fp16 tiles -> HMX proof. One dense tensor, not a whole LLM.
//! No production hooks. Device opening requires the guard/GO handshake.
use std::io::Write;
use std::time::{Duration,Instant};
use oxn::linear::{NpuConfig,NpuCtx,WeightRows};
use oxn::{Backend,tiles,wfmt::WFmt};
use model::npu_rows::GgufRows;

fn half(v:f32)->f32{tiles::half_to_f32(tiles::f32_to_half(v))}
fn input(t:usize,c:usize)->f32{
    let mut h=(t as u32).wrapping_mul(0x9e3779b9)^(c as u32).wrapping_mul(0x85ebca6b)^0x12345678;
    h^=h>>16;h=h.wrapping_mul(0x7feb352d);h^=h>>15;
    ((h&1023)as i32-512)as f32/128.0
}
fn run()->Result<(),String>{
    let args:Vec<String>=std::env::args().skip(1).collect();
    let mut model=None;let mut name=None;let mut backend="emu".to_string();let mut tokens=32usize;let mut go=None;let mut dry=false;
    let mut i=0;
    while i<args.len(){
        if args[i]=="--dry-run"{dry=true;i+=1;continue;}
        if i+1>=args.len(){return Err("option needs a value".into());}
        match args[i].as_str(){
            "--model"=>model=Some(args[i+1].clone()),
            "--tensor"=>name=Some(args[i+1].clone()),
            "--backend"=>backend=args[i+1].clone(),
            "--tokens"=>tokens=args[i+1].parse().map_err(|_|"invalid token count")?,
            "--go-file"=>go=Some(args[i+1].clone()),
            _=>return Err("unknown option".into())
        }i+=2;
    }
    if ![32,128,512,1024].contains(&tokens){return Err("tokens must be32/128/512/1024".into());}
    let name=name.ok_or("--tensor required")?;
    if !name.bytes().all(|c|c.is_ascii_alphanumeric()||b"._-".contains(&c)){return Err("unsafe tensor label".into());}
    let file=gguf::Gguf::open(model.ok_or("--model required")?).map_err(|e|e.to_string())?;
    let tensor=file.header.tensor(&name).ok_or("tensor absent")?;
    let rows=GgufRows::new(tensor,file.tensor_data(tensor))?;
    let(o,k)=rows.shape();
    if o%32!=0||k%32!=0||o>32768||k>32768{return Err("probe shape out of bounds".into());}
    println!("{{\"event\":\"plan\",\"tensor\":\"{name}\",\"source_type\":{},\"tokens\":{tokens},\"o\":{o},\"k\":{k},\"fp16_weight_bytes\":{}}}",tensor.ty.0,o*k*2);
    if dry{return Ok(());}
    let be:Box<dyn Backend>=match backend.as_str(){
        "emu"=>Box::new(oxn::emu::Emu::new(8<<20,1)),
        "device"=>{
            let go=go.ok_or("device needs --go-file")?;
            println!("{{\"event\":\"ready_before_device\",\"pid\":{}}}",std::process::id());
            std::io::stdout().flush().map_err(|e|e.to_string())?;
            let until=Instant::now()+Duration::from_secs(30);
            while !std::path::Path::new(&go).is_file(){
                if Instant::now()>until{return Err("guard handshake timeout; no device opened".into());}
                std::thread::sleep(Duration::from_millis(20));
            }
            #[cfg(feature="npu-device")]{Box::new(oxn::device::Device::open(1)?)}
            #[cfg(not(feature="npu-device"))]{return Err("build with npu-device feature".into());}
        },_=>return Err("backend must be emu/device".into())
    };
    let cfg=NpuConfig{budget_bytes:256<<20,kc_max:0,ycol:true,pack_threads:4,cpu_rows_pct:0,weights:WFmt::Fp16,max_blob_bytes:128<<20,audit:false};
    let mut ctx=NpuCtx::new(be,cfg)?;
    let lin=ctx.add_linear_rows(&name,&rows)?.ok_or("linear did not fit")?;
    let st=ctx.stats();
    println!("{{\"event\":\"packed\",\"seconds\":{:.9},\"workers\":{},\"weights_bytes\":{},\"reserved_dma_bytes\":{},\"parts\":{}}}",st.pack_s,st.pack_threads,st.weights_bytes,st.allocated_bytes,st.parts);
    let act=ctx.act(lin.session,0,tokens,k,lin.kc)?;
    let mut x=vec![0f32;k];
    for t in 0..tokens{
        for(c,v)in x.iter_mut().enumerate(){*v=input(t,c);}
        // SAFETY: exclusive valid token row, complete K, no concurrent backend call.
        unsafe{act.put(t,0,&x);}
    }
    let mut reference_row=vec![0f32;k];
    let mut hash_reference=None;
    // A numerical check runs on the warm-up and after each measured invocation.
    for rep in 0..4{
        ctx.reset_step_stats();let t0=Instant::now();let y=ctx.run(&lin,&act)?;let seconds=t0.elapsed().as_secs_f64();
        let mut se=0f64;let mut sq=0f64;let mut so=0f64;let mut sr=0f64;let mut h=0xcbf29ce484222325u64;
        for t in [0,tokens/2,tokens-1]{
            for sample in 0..32{
                let col=sample*(o-1)/31;
                rows.read_row(col,&mut reference_row)?;
                let mut rounded=0f64;let mut original=0f64;
                for(c,&w)in reference_row.iter().enumerate(){
                    let a=input(t,c);rounded+=half(a)as f64*half(w)as f64;original+=a as f64*w as f64;
                }
                let mut got=[0f32;1];
                // SAFETY: run complete; no later run until this output is checked.
                unsafe{y.row(t,col,&mut got);}
                if !got[0].is_finite(){return Err("nonfinite accelerator output".into());}
                let g=got[0]as f64;se+=(g-rounded).powi(2);sq+=rounded*rounded;so+=(g-original).powi(2);sr+=original*original;
                h=(h^got[0].to_bits()as u64).wrapping_mul(0x100000001b3);
            }
        }
        let rel=(se/sq.max(1e-30)).sqrt();let original_rel=(so/sr.max(1e-30)).sqrt();
        if !rel.is_finite()||rel>0.001{return Err(format!("fp16 reference relative error{rel} exceeds0.001"));}
        if let Some(old)=hash_reference{if old!=h{return Err("repeated output sample hash differs".into());}}else{hash_reference=Some(h);}
        let st=ctx.stats();
        println!("{{\"event\":\"linear_result\",\"rep\":{rep},\"warmup\":{},\"seconds\":{seconds:.9},\"dsp_us\":{},\"backend_calls\":{},\"relative_fp16_f64\":{rel:.9},\"relative_original_f64\":{original_rel:.9},\"sample_hash\":\"{h:016x}\",\"passed\":true}}",rep==0,st.dsp_us,st.calls);
        std::io::stdout().flush().map_err(|e|e.to_string())?;
    }
    drop(ctx);println!("{{\"event\":\"complete\",\"passed\":true}}");Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("npu-linear-proof: STOP {e}");std::process::exit(3);}}
