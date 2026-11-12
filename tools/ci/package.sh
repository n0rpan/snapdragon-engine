#!/usr/bin/env bash
# Package release binaries into one tarball with licences, notices and a README.
#   tools/ci/package.sh NAME VERSION DIST_DIR CARGO_PACKAGE FEATURES TARGET_DIR BIN [BIN...]
# FEATURES is a cargo feature list for the licence scan, or "-" for none.
# Writes DIST_DIR/NAME-VERSION-linux-aarch64.tar.gz and its .sha256.
# Used by .github/workflows/release.yml.
set -euo pipefail
NAME=$1 VERSION=$2 DIST=$3 PKG=$4 FEATURES=$5 TARGET=$6
shift 6
[ $# -ge 1 ] || { echo "no binaries given" >&2; exit 2; }
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STEM="$NAME-$VERSION-linux-aarch64"
STAGE=$(mktemp -d)/$STEM
mkdir -p "$STAGE/bin" "$STAGE/licenses" "$DIST"

for b in "$@"; do install -m 0755 "$TARGET/release/$b" "$STAGE/bin/$b"; done
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$ROOT/THIRD_PARTY_NOTICES" "$STAGE/"
cp "$ROOT/vendor/ggml-subset/LICENSE" "$STAGE/licenses/ggml-subset-LICENSE"
cp "$ROOT/vendor/ggml-flash/LICENSE" "$STAGE/licenses/ggml-flash-LICENSE"
cp "$ROOT/third_party/stable-diffusion.cpp-LICENSE" "$STAGE/licenses/"
feat=()
[ "$FEATURES" = "-" ] || feat=(--features "$FEATURES")
(cd "$ROOT" && python3 -I tools/ci/crate_licenses.py "$PKG" "$STAGE/licenses/crates" "${feat[@]}")

flash_note=""
if [[ "$FEATURES" =~ (^|[[:space:],])(server/)?flash($|[[:space:],]) ]]; then
  : "${OXF_GGML_PROD:?Flash packaging requires the pinned source licence bundle}"
  python3 -I "$ROOT/tools/ci/flash_licenses.py" package "$OXF_GGML_PROD" "$STAGE"
  cp "$OXF_GGML_PROD/MANIFEST" "$STAGE/GGML-PROD-MANIFEST.txt"
  flash_note="
Flash links the production ggml static libraries, built in the same CI run from the
pinned public sources (BigMoeOnEdge 930b2c9 + llama.cpp fork 7ac98f4 + the demand-mapping
patch, tools/ci/build_ggml_prod.sh). Their hashes and build flags are in
GGML-PROD-MANIFEST.txt. Both pinned build-source licences are in licenses/;
FLASH-SOURCE-LICENSES.json records their hashes and upstream NOTICE absence."
fi

# Highest glibc symbol version any binary needs.
glibc=$(for b in "$STAGE"/bin/*; do objdump -T "$b"; done | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -n1)
glibc=${glibc#GLIBC_}
sums=$(cd "$STAGE/bin" && sha256sum -- *)
repo=${GITHUB_REPOSITORY:-n0rpan/snapdragon-engine}

cat > "$STAGE/README.txt" <<EOF
$NAME $VERSION for Linux on ARM64 (aarch64)
$(printf '%*s' $((${#NAME} + ${#VERSION} + 26)) '' | tr ' ' '=')

Built by GitHub Actions from the tagged source of https://github.com/$repo
  source commit: ${GITHUB_SHA:-unknown}
  ref:           ${GITHUB_REF_NAME:-unknown}
  workflow run:  ${GITHUB_SERVER_URL:-https://github.com}/$repo/actions/runs/${GITHUB_RUN_ID:-unknown}
  toolchain:     $(rustc --version), $(cc --version | head -n1)
Nobody builds these binaries by hand; the release workflow is .github/workflows/release.yml.

Binaries (bin/):
$sums

CPU: ARMv8.6-A with dotprod, i8mm, fp16 and bf16, no SVE needed. Tuned for and measured
on the Snapdragon X Elite (X1E80100) only; the X Plus has not been tested. The Rust code is compiled with
target-cpu=oryon-1 and the vendored ggml kernels with
-march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme. Other cores with the same features
(the CI tests run on an Arm Neoverse N2 class runner) are untested. A core without i8mm or
bf16 is reported to stop with "Illegal instruction"; this has not been tested.
OS:  Linux aarch64 with glibc $glibc or newer.
$flash_note
Numerics: the published speed and quality numbers were measured with binaries built on the
test laptop (GCC 16.1). This CI build uses the toolchain above, whose machine code differs in
places, so outputs can differ from those builds in the last bits of floating-point results
(the "reorder" class of the project's correctness rules). The model-free tests and the Flash
selftest pass on it; it has not been through the lab's model-based identity or quality gates.

Usage, models and benchmarks: README.md of the repository and the companion lab repository
https://github.com/n0rpan/snapdragon-local-ai-lab. No model weights are included.

Verify this download:
  sha256sum -c SHA256SUMS                        (from the same release)
  gh attestation verify $STEM.tar.gz --repo $repo \\
    --signer-workflow $repo/.github/workflows/release.yml

Licences: LICENSE (MIT, this project), NOTICE, THIRD_PARTY_NOTICES, licenses/ (vendored
ggml, stable-diffusion.cpp, both pinned Flash build sources where linked, and every
Rust crate in licenses/crates).
EOF

epoch=${SOURCE_DATE_EPOCH:-$(git -C "$ROOT" log -1 --format=%ct 2>/dev/null || date +%s)}
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$epoch" \
    -C "$(dirname "$STAGE")" -czf "$DIST/$STEM.tar.gz" "$STEM"
(cd "$DIST" && sha256sum "$STEM.tar.gz" > "$STEM.tar.gz.sha256")
echo "glibc>=$glibc"
cat "$DIST/$STEM.tar.gz.sha256"
