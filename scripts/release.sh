#!/usr/bin/env bash
# Release helper — bump the versions, roll the changelog, commit, tag, and (after
# a prompt) push, which triggers the coordinated release workflow.
#
#   Usage:  scripts/release.sh X.Y.Z
#   Windows: powershell -File scripts/release.ps1 X.Y.Z
#
# It bakes in the "bump the manifests BEFORE tagging" rule, so the tag always
# matches Cargo.toml — the release workflow rejects a mismatch. Uses GNU
# sed (Git Bash / Linux).
set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

version="${1:-}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "usage: $0 X.Y.Z"
tag="v$version"

cd "$(git rev-parse --show-toplevel)"

# --- preflight ---
[[ -z "$(git status --porcelain)" ]] || die "working tree is not clean"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "main" ]] || die "not on main"
git rev-parse -q --verify "refs/tags/$tag" >/dev/null && die "tag $tag already exists"

echo "==> releasing $version"

release_files=(Cargo.toml Cargo.lock pyproject.toml CHANGELOG.md index.html guide.html docs/index.html tests/python/test_frame_and_profile.py)
committed=0
cleanup() {
  if [[ "$committed" == 0 ]]; then
    git restore --staged --worktree -- "${release_files[@]}" 2>/dev/null || true
  fi
}
trap cleanup ERR INT TERM

# --- bump the package version (only the first top-level `version = ` line, so
#     dependency pins like { version = "=0.6.10" } are left alone) ---
python scripts/sync-version.py "$version"

# --- roll the changelog: stamp [Unreleased] as this version, keep a fresh one ---
sed -i -E "s/^## \[Unreleased\]/## [Unreleased]\n\n## [$version] - $(date +%F)/" CHANGELOG.md

# --- sync Cargo.lock, then verify the crate still packages cleanly ---
cargo build -q
echo "==> verifying package (cargo publish --dry-run)"
cargo publish --dry-run --locked --allow-dirty >/dev/null

# --- commit + tag ---
git add "${release_files[@]}"
git commit -q -m "Release $version"
git tag "$tag"
committed=1
echo "==> committed and tagged $tag"

# --- push (this publishes) ---
read -r -p "Push main + $tag now? This publishes to crates.io and PyPI. [y/N] " reply
if [[ "$reply" == "y" || "$reply" == "Y" ]]; then
  git push origin main "$tag"
  echo "==> pushed — watch the release workflow."
else
  echo "not pushed. When ready:  git push origin main $tag"
fi
