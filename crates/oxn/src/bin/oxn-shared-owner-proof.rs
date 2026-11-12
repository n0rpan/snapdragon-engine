//! Bounded W2b lifetime/coherence proof. Old skel, 16 MiB synthetic CPU weights.
//! No model, no new arithmetic, no deliberately invalid device operations.
use std::io::Write;
use std::time::{Duration, Instant};
const BYTES: usize = 16 << 20;
fn diagonal(bytes: &mut [u8], value: f32) {
    for off in [0, bytes.len() - 2048] {
        bytes[off..off + 2048].fill(0);
        for i in 0..32 {
            let p = off + oxn::tiles::wt_elem_idx(i, i) * 2;
            bytes[p..p + 2].copy_from_slice(&oxn::tiles::f32_to_half(value).to_le_bytes());
        }
    }
}
fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &v| {
        (h ^ v as u64).wrapping_mul(0x100000001b3)
    })
}
fn emit(event: &str, details: &str) -> Result<(), String> {
    println!("{{\"event\":\"{event}\",{details}}}");
    std::io::stdout().flush().map_err(|e| e.to_string())
}
#[cfg(feature = "device")]
fn xy(be: &mut oxn::device::Device) -> Result<(oxn::Buf, oxn::Buf), String> {
    use oxn::{
        tiles::{self, ActLayout},
        Backend,
    };
    let x = be.alloc(0, 4096)?;
    let y = be.alloc(0, 4096)?;
    for row in 0..32 {
        let mut a = [0.0; 32];
        a[row] = 1.0;
        // SAFETY: X is an exclusive live device-owned 4096-byte buffer.
        unsafe {
            tiles::write_act_row(x.ptr.cast(), &ActLayout::new(32, 32, 32), row, 0, &a);
        }
    }
    Ok((x, y))
}
#[cfg(feature = "device")]
fn verify(
    be: &mut oxn::device::Device,
    x: oxn::Buf,
    y: oxn::Buf,
    weight: &oxn::ImportedWeight,
    expected: f32,
) -> Result<(), String> {
    let w = weight.buf();
    // SAFETY: backend and owner retain every live allocation; W is sealed,
    // CPU-immutable; GEMM completes synchronously and writes only exclusive Y.
    unsafe {
        oxn::mapping_probe::verify_diagonal(be, 0, x, y, w, 0, expected)?;
        oxn::mapping_probe::verify_diagonal(be, 0, x, y, w, w.len - 2048, expected)?;
    }
    Ok(())
}
#[cfg(feature = "device")]
fn run(go: &str) -> Result<(), String> {
    use oxn::{device::Device, shared_memory::SharedAllocator, Backend};
    if go.is_empty() {
        return Err("mandatory --go-file missing; no allocation/device opened".into());
    }
    emit(
        "ready_before_device",
        &format!("\"pid\":{}", std::process::id()),
    )?;
    let until = Instant::now() + Duration::from_secs(30);
    while !std::path::Path::new(go).is_file() {
        if Instant::now() > until {
            return Err("guard handshake timeout; no allocation/device opened".into());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let allocator = SharedAllocator::open(BYTES)?;
    let mut cpu = allocator.allocate(BYTES)?;
    diagonal(cpu.as_mut_bytes(), 1.0);
    let cpu = cpu.seal();
    let id = cpu.id();
    let ptr = cpu.as_bytes().as_ptr();
    let before = checksum(cpu.as_bytes());
    emit(
        "cpu_owner_ready",
        &format!(
            "\"cpu_owned_bytes\":{},\"allocation_id\":{id},\"no_dsp_yet\":true",
            allocator.owned_bytes()
        ),
    )?;
    let mut be = Device::open(1)?;
    be.set_window_limit(32 << 20)?;
    if be.window_bytes(0)? != 0 {
        return Err("new backend has nonzero window accounting".into());
    }
    let (x, y) = xy(&mut be)?;
    let imported = be.import_weight(0, &cpu)?;
    let first_slot = imported.buf().slot;
    if be.window_bytes(0)? != BYTES + 8192 {
        return Err("mapped view accounting differs from CPU ownership".into());
    }
    if be.alloc(0, 32 << 20).is_ok() || be.failed() {
        return Err("window budget refused too late or poisoned backend".into());
    }
    if be.import_weight(0, &cpu).is_ok() || be.failed() {
        return Err("duplicate import refused too late or poisoned backend".into());
    }
    let cpu = cpu
        .try_into_mut()
        .err()
        .ok_or("import failed to retain immutable CPU owner")?;
    verify(&mut be, x, y, &imported, 1.0)?;
    be.release_weight(imported)?;
    if be.window_bytes(0)? != 8192 {
        return Err("released weight remained in mapped view ledger".into());
    }
    emit("window_budget_proved","\"limit_bytes\":33554432,\"after_release_bytes\":8192,\"local_refusals_no_device_error\":true")?;
    std::thread::sleep(Duration::from_millis(20));
    if checksum(cpu.as_bytes()) != before {
        return Err("DSP changed immutable CPU weights".into());
    }
    let mut cpu = cpu
        .try_into_mut()
        .map_err(|_| "confirmed release did not relinquish CPU owner")?;
    diagonal(cpu.as_mut_bytes(), 2.0);
    let cpu = cpu.seal();
    if cpu.id() != id || cpu.as_bytes().as_ptr() != ptr {
        return Err("CPU allocation identity changed".into());
    }
    let after = checksum(cpu.as_bytes());
    let imported = be.import_weight(0, &cpu)?;
    if imported.buf().slot != first_slot {
        return Err("released import slot was not reused".into());
    }
    verify(&mut be, x, y, &imported, 2.0)?;
    // Intentionally close with a live import: normal teardown must detach it,
    // while CPU weights remain owned and readable independently of the session.
    be.close()?;
    drop(imported);
    if checksum(cpu.as_bytes()) != after || allocator.owned_bytes() != BYTES {
        return Err("backend close invalidated CPU weights/accounting".into());
    }
    emit("backend_closed_cpu_alive",&format!("\"cpu_owned_bytes\":{BYTES},\"pointer_stable\":true,\"first_last_tiles\":true,\"slot_reused\":true"))?;
    std::thread::sleep(Duration::from_millis(100));
    let mut be = Device::open(1)?;
    be.set_window_limit(32 << 20)?;
    if be.window_bytes(0)? != 0 {
        return Err("new backend has nonzero window accounting".into());
    }
    let (x, y) = xy(&mut be)?;
    let imported = be.import_weight(0, &cpu)?;
    verify(&mut be, x, y, &imported, 2.0)?;
    be.release_weight(imported)?;
    be.close()?;
    if checksum(cpu.as_bytes()) != after {
        return Err("reopened backend changed CPU weights".into());
    }
    let cpu = cpu
        .try_into_mut()
        .map_err(|_| "reopened backend retained CPU owner after confirmed close")?;
    drop(cpu);
    if allocator.owned_bytes() != 0 {
        return Err("CPU-owned bytes did not return to zero".into());
    }
    emit(
        "complete",
        "\"passed\":true,\"cpu_owned_bytes\":0,\"backend_reopen\":true,\"old_skel\":true",
    )
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|s| s == "--dry-run") {
        println!("{{\"kind\":\"CPU owner independent of DSP windows\",\"cpu_dma_bytes\":{BYTES},\"device_owned_peak_bytes\":8192,\"sessions_at_once\":1,\"backend_opens\":2,\"new_skel\":false,\"model\":false}}");
        return;
    }
    #[cfg(feature = "device")]
    {
        let go = args
            .windows(2)
            .find(|p| p[0] == "--go-file")
            .map_or("", |p| p[1].as_str());
        if let Err(e) = run(go) {
            eprintln!("oxn-shared-owner-proof: STOP {e}");
            std::process::exit(3);
        }
    }
    #[cfg(not(feature = "device"))]
    {
        eprintln!("device feature required");
        std::process::exit(2);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_cpu_tiles_have_exact_values_and_middle_stays_untouched() {
        let mut b = vec![0; 16384];
        for v in [1.0, 2.0] {
            diagonal(&mut b, v);
            for off in [0, 14336] {
                for row in 0..32 {
                    for col in 0..32 {
                        let p = off + oxn::tiles::wt_elem_idx(row, col) * 2;
                        assert_eq!(
                            u16::from_le_bytes([b[p], b[p + 1]]),
                            oxn::tiles::f32_to_half(if row == col { v } else { 0.0 })
                        );
                    }
                }
            }
            assert!(b[2048..14336].iter().all(|&v| v == 0));
        }
    }
}
