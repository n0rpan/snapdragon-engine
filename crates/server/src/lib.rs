//! oxe-server: an OpenAI-compatible HTTP server shaped like llama-server, over the
//! [`engine::Engine`] trait (real engine: crates/model once M2 lands; [`engine::StubEngine`]
//! until then). See SERVER.md.

pub mod api;
pub mod cache;
pub mod engine;
#[cfg(feature = "flash")]
pub mod flash_engine;
pub mod flash_switches;
pub mod generate;
pub mod http;
pub mod lookup;
pub mod oxe_engine;
pub mod prefix_store;
pub mod sampling;
pub mod service;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_flash_prod;
#[cfg(test)]
mod tests_hotfix;
#[cfg(test)]
mod tests_flash_prefix;
