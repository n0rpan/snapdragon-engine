//! Prints the klein window plan for the fixture table. Planning arithmetic only; no device access.
//! Usage: plan_klein [--json] [tokens...]   (default tokens: 1536 4608)

use image_npu::plan::{klein_linear_order, parse_tensor_table, plan_json, plan_report, plan_windows, ActReserve, PlanConfig};

const FIXTURE: &str = include_str!("../../../../tests/fixtures/klein-4b-q8.tensors.txt");

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json = args.iter().any(|a| a == "--json");
    let mut tokens: Vec<usize> = args.iter().filter_map(|a| a.parse().ok()).collect();
    if tokens.is_empty() {
        tokens = vec![1536, 4608];
    }
    let rows = parse_tensor_table(FIXTURE);
    let order = klein_linear_order(&rows).expect("klein order");
    for &n in &tokens {
        for act in [ActReserve::Sum, ActReserve::PeakPerLinear] {
            for floor in [10.8, 0.5] {
                let mut cfg = PlanConfig::klein_default(n);
                cfg.act_reserve = act;
                cfg.floor_ms = floor;
                let windows = plan_windows(&order, &cfg).expect("plan");
                if json {
                    println!("{}", plan_json(&windows, &cfg));
                } else {
                    println!("==== tokens={n} act={act:?} floor_ms={floor} ====");
                    print!("{}", plan_report(&windows, &cfg));
                    println!();
                }
            }
        }
    }
}
