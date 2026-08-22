# Contributing to Millwright

Thanks for your interest! Millwright assembles proven Rust crates into one
composable ML lifecycle. Bug reports, small fixes, and focused features are all
welcome.

## Getting set up

```bash
git clone https://github.com/mi7plus/millwright
cd millwright
cargo test              # default features
```

Rust **1.95+** is required (the floor is dictated by transitive engine deps).

## Before you open a PR

The CI is a feature matrix, so run the same checks locally:

```bash
cargo fmt --all --check
cargo clippy --features full --all-targets -- -D warnings
cargo test --features full
cargo test --no-default-features            # the bare-core build must pass too
cargo doc --no-deps --features full         # with RUSTDOCFLAGS="-D warnings"
```

- **Every capability is a cargo feature.** New functionality behind a young
  single-author crate goes behind its own feature; the core stays lean. See the
  feature list in `Cargo.toml`.
- **Engines are pinned to exact versions** (`=x.y.z`) and `Cargo.lock` is
  committed — bump them deliberately, one line, one commit.
- **Add tests.** Prefer a `#[test]` next to the code; lock numeric behaviour in
  `tests/golden.rs` when it matters. Feature-gate tests that need a backend.
- **Run the examples** you touch: `cargo run --features full --example <name>`.

## Try it out

- Examples: [`examples/`](examples) — one runnable program per feature group.
- Benchmarks: `cargo bench --features smartcore-backend`.
- Docs/tutorial: <https://millwright-rs.dev/docs/>.

## Releasing

Maintainers: bump `Cargo.toml` + `pyproject.toml`, roll `CHANGELOG.md`, and tag
`vX.Y.Z` — the workflows publish to crates.io and PyPI via OIDC. The helper
`scripts/release.sh X.Y.Z` does the mechanical steps. See
[`RELEASING.md`](RELEASING.md).

## Scope & conduct

Millwright is a thin facade over proven engines — it orchestrates, it doesn't
reimplement numerics. Please keep PRs focused, and be kind and constructive in
issues and reviews.
