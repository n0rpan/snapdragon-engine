#!/usr/bin/env bash
# CI toolchain inside the Fedora 44 container (GCC 16, the compiler the engine's bit-exactness
# was established with) plus a pinned Rust from rustup. Prints every version it installs.
# The dnf packages are not pinned to exact builds: Fedora mirrors keep only the newest update, so
# a pinned build would stop resolving. The versions are printed here and written to every
# tarball's README.txt instead.
#   tools/ci/setup_toolchain.sh RUST_VERSION
set -euo pipefail
RUST_VERSION=${1:?usage: setup_toolchain.sh RUST_VERSION}
dnf -y -q install --setopt=install_weak_deps=False \
  gcc gcc-c++ binutils make cmake ninja-build git-core python3 \
  tar gzip findutils which diffutils curl ca-certificates
# rustup-init pinned by version and SHA-256 (no curl | sh). The Rust toolchain itself is pinned by
# version; rustup checks its components against the signed channel manifest.
RUSTUP_VERSION=1.29.2
RUSTUP_SHA256=a9c168dbea81a84671c6b7a5c1795cb0eb699020110d107ddfca427bd797e697
tmp=$(mktemp -d)
curl --proto '=https' --tlsv1.2 -sSf -o "$tmp/rustup-init" \
  "https://static.rust-lang.org/rustup/archive/$RUSTUP_VERSION/aarch64-unknown-linux-gnu/rustup-init"
echo "$RUSTUP_SHA256  $tmp/rustup-init" | sha256sum -c -
chmod +x "$tmp/rustup-init"
# rustup warns that the container's HOME (/github/home) is not root's passwd home: expected here.
RUSTUP_INIT_SKIP_PATH_CHECK=yes "$tmp/rustup-init" -y -q --no-modify-path --profile minimal \
  --default-toolchain "$RUST_VERSION"
rm -rf "$tmp"
echo "$HOME/.cargo/bin" >> "${GITHUB_PATH:-/dev/null}"
export PATH="$HOME/.cargo/bin:$PATH"
# actions/checkout runs as another uid than the container's root: let git read the checkout
# (build.rs records the engine commit).
git config --global --add safe.directory "${GITHUB_WORKSPACE:-$PWD}"
cat /etc/fedora-release
cc --version | head -n1
c++ --version | head -n1
ld --version | head -n1
cmake --version | head -n1
echo "ninja $(ninja --version)"
rustc --version
cargo --version
echo "CPU: $(grep -m1 -E '^(CPU part|model name)' /proc/cpuinfo || true), $(nproc) threads"
grep -m1 '^Features' /proc/cpuinfo || true
free -g || true
echo "RLIMIT_MEMLOCK: $(ulimit -l)"
