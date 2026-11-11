//! qwen35moe model side of the engine: descriptor, file-backed repacked weights, static
//! plan, and the decode engine with llama.cpp-exact math.

pub mod desc;
pub mod engine;
pub mod load;
pub mod mtp;
pub mod plan;
pub mod weights;

#[cfg(test)]
mod tests;
