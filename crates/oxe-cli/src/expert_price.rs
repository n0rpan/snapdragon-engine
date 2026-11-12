//! Bounded one-layer routed-expert pricing, not a model hook or quality claim.
//! Original native packed weights are copied once into shared DMA allocations;
//! both controls read those same immutable bytes. All packing/conversion and
//! SwiGLU/scatter costs are in the timed phase. No resident fp16 weight copy.
use super::*;
use model::weights::WeightRef;
use oxn::{Backend,Buf};
use oxn::device::Device;
use oxn::tiles::{ActLayout,OutLayout,write_act_row,read_out_row};
use sched::Team;

fn parallel(pool:&Pool,n:usize,job:impl Fn(usize)+Sync){
 pool.run_job(&|ctx:&mut sched::Ctx|{for i in (ctx.tid..n).step_by(ctx.nthreads){job(i);}});
}
fn quant(pool:&Pool,x:&[f32],rows:usize,k:usize,q:kernels_sys::QType)->Vec<u8>{
 let rb=q.act().bytes(k);let mut a=vec![0u8;rows*rb];let p=a.as_mut_ptr() as usize;
 parallel(pool,rows,|r|unsafe{kernels_sys::quantize_act(q.act(),&x[r*k..(r+1)*k],std::slice::from_raw_parts_mut((p+r*rb)as*mut u8,rb));});a
}
fn cpu_mmid(pool:&Pool,weights:&[WeightRef],x:&[f32],n:usize,unique:bool)->Vec<Vec<f32>>{
 let w=weights[0];let k=w.k;let o=w.rows;let e=w.n_expert;
 let a=quant(pool,x,if unique{n*32}else{e*n},k,w.qtype());let rb=w.qtype().act().bytes(k);
 let mut ys:Vec<Vec<f32>>=weights.iter().map(|_|vec![0.;e*n*o]).collect();
 let yp:Vec<usize>=ys.iter_mut().map(|v|v.as_mut_ptr()as usize).collect();
 let nc=o.div_ceil(128);let jobs=e*nc;
 pool.run_job(&|ctx:&mut sched::Ctx|{
  let mut pack=vec![0u32;n.div_ceil(2)*(k/256)*kernels_sys::PF_XBLK_BYTES/4];
  for job in (ctx.tid..jobs).step_by(ctx.nthreads){
   let expert=job/nc;let c0=job%nc*128;let cols=128.min(o-c0);
   let rows:Vec<*const u8>=(0..n).map(|m|a.as_ptr().wrapping_add((if unique{expert/8+32*m}else{expert*n+m})*rb)).collect();
   unsafe{kernels_sys::mmid_tile_pack_raw(&rows,k,pack.as_mut_ptr()as*mut u8);}
   for (wi,w) in weights.iter().enumerate(){
    let out:Vec<*mut f32>=(0..n).map(|m|(yp[wi]+((expert*n+m)*o+c0)*4)as*mut f32).collect();
    // Disjoint expert/output-column units; each thread's packed scratch is private.
    assert!(unsafe{kernels_sys::mmid_tile_raw(w.qtype(),k,&out,w.rows_ptr(expert,c0),pack.as_ptr()as*const u8,cols)});
   }
  }
 });ys
}
fn swiglu(pool:&Pool,g:&[f32],u:&[f32])->Vec<f32>{
 let mut z=vec![0.;g.len()];let p=z.as_mut_ptr()as usize;
 parallel(pool,g.len().div_ceil(2048),|b|{for i in b*2048..((b+1)*2048).min(g.len()){
  unsafe{*((p+4*i)as*mut f32)=g[i]/(1.+(-g[i]).exp())*u[i];}
 }});z
}
fn scatter(pool:&Pool,y:&[f32],n:usize,k:usize)->Vec<f32>{
 let mut out=vec![0.;n*32*k];let p=out.as_mut_ptr()as usize;
 parallel(pool,n*32,|token|{let m=token/32;let e0=(token%32)*8;
  for c in 0..k{let mut sum=0.;for j in 0..8{sum+=y[((e0+j)*n+m)*k+c]*0.125;}
   unsafe{*((p+(token*k+c)*4)as*mut f32)=sum;}
  }
 });out
}
fn pack(pool:&Pool,x:&[f32],dst:Buf,n:usize,k:usize,unique:bool){
 let t=n.max(32);let lay=ActLayout::new(t,k,k);let ptr=dst.ptr as usize;
 assert!(256*lay.bytes()<=dst.len);
 parallel(pool,256,|e|{let p=(ptr+e*lay.bytes())as*mut u16;
  // Padding participates in the real kernel traffic/work, but not useful FLOPs.
  if n<t{unsafe{std::ptr::write_bytes(p,0,lay.bytes()/2);}}
  for m in 0..n{let row=if unique{e/8+32*m}else{e*n+m};
   unsafe{write_act_row(p,&lay,m,0,&x[row*k..(row+1)*k]);}
  }
 });
}
fn unpack(pool:&Pool,src:Buf,n:usize,o:usize)->Vec<f32>{
 let t=n.max(32);let lay=OutLayout{t_pad:t,o,nchunks:1,ycol:true};let ptr=src.ptr as usize;
 assert!(256*lay.bytes()<=src.len);let mut y=vec![0.;256*n*o];let yp=y.as_mut_ptr()as usize;
 parallel(pool,256*n,|row|{let e=row/n;let m=row%n;
  unsafe{read_out_row((ptr+e*lay.bytes())as*const u16,&lay,m,0,std::slice::from_raw_parts_mut((yp+row*o*4)as*mut f32,o));}
 });y
}
fn kind(w:WeightRef)->Result<u32,String>{match w.qtype(){kernels_sys::QType::Q3K=>Ok(3),kernels_sys::QType::Q4K=>Ok(4),_=>Err("native price supports Q3_K/Q4_K only".into())}}

fn proof(w:&[u8],x:Buf,y:Buf,n:usize,k:usize,o:usize,kind:u32)->Result<serde_json::Value,String>{
 let t=n.max(32);let xt=unsafe{std::slice::from_raw_parts(x.ptr as*const u16,256*t*k)};
 let yt=unsafe{std::slice::from_raw_parts(y.ptr as*const u16,256*t*o)};
 let mut sq=0.;let mut base=0.;let mut worst=0f64;
 for i in 0..64usize{
  let e=i*73%256;let m=i*17%n;let c=i*173%o;
  let want=oxn::device::native_reference_sample(w,xt,kind,t as u32,k as u32,o as u32,256,e as u32,m as u32,c as u32)?;
  let at=e*t*o+((c/32)*(t/32)+m/32)*1024+oxn::tiles::out_elem_idx(m%32,c%32);
  let got=oxn::tiles::half_to_f32(yt[at])as f64;
  if !got.is_finite(){return Err("nonfinite real-weight device sample".into());}
  let d=got-want;sq+=d*d;base+=want*want;worst=worst.max(d.abs());
 }
 let rel=(sq/base.max(1e-30)).sqrt();if rel>6e-4{return Err(format!("real-weight device f64 sample RMSE {rel} >6e-4"));}
 Ok(json!({"positions":64,"rel_rmse":rel,"max_abs":worst,"reference":"f64 on original native weights and actual fp16 input tiles; local projection only"}))
}

pub(super) fn run(a:&Args)->Result<(),String>{
 let s=setup(a,512);let layer=a.num("layer",5);let reps=a.num("reps",3);
 if layer>=s.loaded.plan.layers.len()||!(1..=3).contains(&reps){return Err("layer or reps out of pricing bounds".into());}
 let l=&s.loaded.plan.layers[layer];
 let ws=[*s.loaded.ws.get(l.gate_exps),*s.loaded.ws.get(l.up_exps),*s.loaded.ws.get(l.down_exps)];
 let k=ws[0].k;let o=ws[0].rows;
 if k!=2048||o!=512||ws.iter().any(|w|w.n_expert!=256)||ws[1].k!=k||ws[1].rows!=o||ws[2].k!=o||ws[2].rows!=k{return Err("one Ornith layer 2048/512/256 required".into());}
 for &w in &ws{kind(w)?;if !kernels_sys::mmid_tile_has(w.qtype()){return Err("CPU pair-tile unavailable".into());}}
 let go=a.get("npu-go-file").ok_or("guard go-file required")?;
 println!("{}",json!({"event":"ready_before_device","pid":std::process::id()}));std::io::stdout().flush().map_err(|e|e.to_string())?;
 let until=Instant::now()+Duration::from_secs(30);while !Path::new(go).is_file(){if Instant::now()>until{return Err("guard handshake timeout; unopened".into());}std::thread::sleep(Duration::from_millis(20));}
 let allocator=oxn::shared_memory::SharedAllocator::open(1<<30)?;let mut owners=Vec::new();let mut cpu=ws;
 let start=Instant::now();
 for (i,w) in ws.iter().enumerate(){
  let mut allocation=allocator.allocate(w.bytes.div_ceil(4096)*4096)?;let p=allocation.as_mut_bytes().as_mut_ptr()as usize;
  parallel(&s.pool,w.bytes.div_ceil(1<<20),|b|{let at=b*(1<<20);let len=(1<<20).min(w.bytes-at);unsafe{std::ptr::copy_nonoverlapping((w.ptr+at)as*const u8,(p+at)as*mut u8,len);}});
  let memory=allocation.seal();if memory.as_bytes()[..w.bytes]!=unsafe{std::slice::from_raw_parts(w.ptr as*const u8,w.bytes)}[..]{return Err("native copy mismatch".into());}
  cpu[i].ptr=memory.as_bytes().as_ptr()as usize;owners.push(memory);
 }
 let copy_seconds=start.elapsed().as_secs_f64();let mut device=Device::open(1)?;device.set_window_limit(3usize<<30)?;
 let start=Instant::now();let mut imported=Vec::new();for w in &owners{imported.push(device.import_weight(0,w)?);}
 // Max geometry N128. Every scratch allocation is reused; total exact DMA logged.
 let xbuf=device.alloc(0,256*128*k*2)?;let zbuf=device.alloc(0,256*128*o*2)?;
 let ybuf=device.alloc(0,256*128*k*2)?;
 let mapping_seconds=start.elapsed().as_secs_f64();let dma=allocator.owned_bytes()+xbuf.len+zbuf.len+ybuf.len;
 if dma>1<<30{return Err("pricing DMA exceeds 1GiB cap".into());}
 println!("{}",json!({"event":"pricing_ready","model_sha256":s.loaded.ws.info.model_sha256,"layer":layer,"cpus":s.pool.cpus(),"copy_seconds":copy_seconds,"mapping_seconds":mapping_seconds,"weight_bytes":ws.map(|w|w.bytes),"exact_dma_bytes":dma,"mapped_bytes":device.window_bytes(0)?,"balanced_routing":true,"top_k":8,"full_model_quality_qualified":false}));
 let mut rows=Vec::new();
 for n in [16usize,128]{
  let x:Vec<f32>=(0..n*32*k).map(|i|((i*73%1021)as f32-510.)/510.).collect();
  // One warm-up each, then mirrored order CPU/NPU/NPU/CPU/CPU/NPU (3/arm).
  let arms=[false,true,false,true,true,false,false,true];
  let mut hashes:HashMap<bool,String>=HashMap::new();
  for (index,npu) in arms.into_iter().enumerate().take(2+2*reps){
   let begin=Instant::now();let mut stats=Vec::new();let mut phases=BTreeMap::new();let out;
   if !npu{
    let stamp=Instant::now();let gu=cpu_mmid(&s.pool,&cpu[..2],&x,n,true);phases.insert("gate_up",stamp.elapsed().as_secs_f64());
    let stamp=Instant::now();let z=swiglu(&s.pool,&gu[0],&gu[1]);phases.insert("swiglu",stamp.elapsed().as_secs_f64());
    let stamp=Instant::now();let down=cpu_mmid(&s.pool,&cpu[2..],&z,n,false);phases.insert("down",stamp.elapsed().as_secs_f64());
    let stamp=Instant::now();out=scatter(&s.pool,&down[0],n,k);phases.insert("scatter",stamp.elapsed().as_secs_f64());
   }else{
    let stamp=Instant::now();pack(&s.pool,&x,xbuf,n,k,true);phases.insert("input_gather_pack",stamp.elapsed().as_secs_f64());
    let stamp=Instant::now();
    stats.push(device.native_batch(xbuf,imported[0].buf(),zbuf,n.max(32)as u32,k as u32,o as u32,256,kind(cpu[0])?)?);
    let g=unpack(&s.pool,zbuf,n,o);
    if index==1{println!("{}",json!({"event":"device_sample_proof","n":n,"projection":"gate","result":proof(owners[0].as_bytes(),xbuf,zbuf,n,k,o,kind(cpu[0])?)?}));}
    stats.push(device.native_batch(xbuf,imported[1].buf(),zbuf,n.max(32)as u32,k as u32,o as u32,256,kind(cpu[1])?)?);
    let u=unpack(&s.pool,zbuf,n,o);
    if index==1{println!("{}",json!({"event":"device_sample_proof","n":n,"projection":"up","result":proof(owners[1].as_bytes(),xbuf,zbuf,n,k,o,kind(cpu[1])?)?}));}
    phases.insert("gate_up_rpc_unpack",stamp.elapsed().as_secs_f64());
    let stamp=Instant::now();let z=swiglu(&s.pool,&g,&u);pack(&s.pool,&z,zbuf,n,o,false);phases.insert("swiglu_down_pack",stamp.elapsed().as_secs_f64());
    let stamp=Instant::now();
    stats.push(device.native_batch(zbuf,imported[2].buf(),ybuf,n.max(32)as u32,o as u32,k as u32,256,kind(cpu[2])?)?);
    let down=unpack(&s.pool,ybuf,n,k);
    if index==1{println!("{}",json!({"event":"device_sample_proof","n":n,"projection":"down","result":proof(owners[2].as_bytes(),zbuf,ybuf,n,o,k,kind(cpu[2])?)?}));}
    phases.insert("down_rpc_unpack",stamp.elapsed().as_secs_f64());
    let stamp=Instant::now();out=scatter(&s.pool,&down,n,k);phases.insert("scatter",stamp.elapsed().as_secs_f64());
   }
   let seconds=begin.elapsed().as_secs_f64();if out.iter().any(|v|!v.is_finite()){return Err("nonfinite projection phase; discard".into());}
   let useful_flops=6f64*256.*n as f64*k as f64*o as f64;
   let hash=format!("{:016x}",model::weights::hash_bytes(unsafe{std::slice::from_raw_parts(out.as_ptr()as*const u8,out.len()*4)}));
   if let Some(old)=hashes.get(&npu){if old!=&hash{return Err("same-arm output changed across repeats".into());}}else{hashes.insert(npu,hash.clone());}
   let row=json!({"event":"expert_phase_result","n_per_expert":n,"tokens":n*32,"index":index,"arm":if npu{"NPU"}else{"CPU"},"warmup":index<2,"seconds":seconds,"phases":phases,"useful_TOPS":useful_flops/seconds/1e12,"output_hash":hash,"nonfinite":0,"dsp":stats.iter().map(|s|json!({"wall_seconds":s.wall_s,"dsp_us":s.dsp_us,"cycles":s.pcycles,"wait_cycles":s.wait_pcycles,"hmx_cycles":s.hmx_pcycles,"dequant_cycles":s.deq_pcycles,"steps":s.steps})).collect::<Vec<_>>()});
   println!("{row}");std::io::stdout().flush().map_err(|e|e.to_string())?;rows.push(row);
  }
 }
 device.close()?;drop(owners);assert_eq!(allocator.owned_bytes(),0);
 let result=json!({"passed":true,"kind":"bounded one-layer uniform-routing pricing, not whole-model quality or performance","rows":rows,"exact_dma_bytes":dma,"copy_seconds":copy_seconds,"mapping_seconds":mapping_seconds});
 if let Some(p)=a.get("out"){std::fs::write(p,serde_json::to_vec_pretty(&result).unwrap()).map_err(|e|e.to_string())?;}
 println!("{}",json!({"event":"complete","passed":true,"owned_bytes_after_close":allocator.owned_bytes()}));Ok(())
}
