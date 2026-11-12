#!/usr/bin/env bash
# Self-hosted helper prep (ci-arm64.yml): pinned Rust via the runner user's rustup, versions, and
# the disk bound. Wipes the shared target cache when it exceeds TARGET_CACHE_MAX_GIB, and fails
# when the home filesystem would drop below MIN_FREE_GIB. No sudo, nothing installed system-wide.
set -euo pipefail
: "${RUST_VERSION:?}" "${CARGO_TARGET_DIR:?}"
max=${TARGET_CACHE_MAX_GIB:-25} min=${MIN_FREE_GIB:-20}
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
echo "$HOME/.cargo/bin" >> "${GITHUB_PATH:-/dev/null}"
echo "$HOME/.local/bin" >> "${GITHUB_PATH:-/dev/null}"
rustup toolchain list | grep -q "^$RUST_VERSION" || rustup toolchain install --profile minimal "$RUST_VERSION"
mkdir -p "$CARGO_TARGET_DIR"
used=$(du -s --block-size=1G "$CARGO_TARGET_DIR" | cut -f1)
if [ "$used" -gt "$max" ]; then echo "target cache ${used} GiB > ${max} GiB: wiping"; rm -rf "$CARGO_TARGET_DIR"; mkdir -p "$CARGO_TARGET_DIR"; fi
free=$(df --output=avail --block-size=1G "$CARGO_TARGET_DIR" | tail -n1 | tr -d ' ')
echo "target cache ${used} GiB (max ${max}), free ${free} GiB (min ${min})"
[ "$free" -ge "$min" ] || { echo "free space ${free} GiB below ${min} GiB: refusing to build"; exit 1; }
cc --version | head -n1
command -v cmake >/dev/null && cmake --version | head -n1 || echo "cmake: none (Flash job needs it)"
command -v ninja >/dev/null && echo "ninja $(ninja --version)" || echo "ninja: none"
rustc --version
cargo --version
echo "CPU: $(grep -m1 -E '^(CPU part|model name)' /proc/cpuinfo || true), $(nproc) threads"
grep -m1 '^Features' /proc/cpuinfo || true
