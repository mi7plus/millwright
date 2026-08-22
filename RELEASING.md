# Releasing

Millwright ships as one Rust crate (crates.io) and one Python wheel (PyPI), from
the same source.

## Preflight

1. `main` is green in CI and `CHANGELOG.md`'s `[Unreleased]` section is current.
2. Bump the version in **both** `Cargo.toml` and `pyproject.toml`, move the
   changelog's `[Unreleased]` entries under the new version, and commit.
3. Prove the packaged crate builds exactly as crates.io will build it:

   ```bash
   cargo publish --dry-run --locked
   ```

## Rust crate → crates.io

Requires a crates.io API token once (`cargo login <token>`), then:

```bash
cargo publish --locked
```

docs.rs builds the documentation automatically with the `full` feature set (see
`[package.metadata.docs.rs]` in `Cargo.toml`); the `python` feature is excluded
there because it needs a Python interpreter.

## Python wheel → PyPI

Built by [maturin](https://www.maturin.rs/) from the `python` feature:

```bash
maturin build --release          # wheels land in target/wheels/
maturin publish                  # needs a PyPI token
```

## Tag

```bash
git tag vX.Y.Z && git push --tags
```

## Notes

- The crate pins its engines to exact versions and commits `Cargo.lock`, so a
  publish is fully reproducible.
- MSRV is declared in `Cargo.toml` (`rust-version`) and enforced by cargo for
  consumers; it is dictated by transitive engine deps, so expect it to rise over
  time.
