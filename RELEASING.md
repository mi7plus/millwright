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

Automated by the **`release-crate.yml`** workflow, which publishes on a `v*` tag
via **Trusted Publishing (OIDC)** — no token is stored anywhere. One-time setup,
on crates.io (crate → Settings → Trusted Publishing → add GitHub):

| field               | value              |
| ------------------- | ------------------ |
| Repository owner    | `mi7plus`          |
| Repository name     | `millwright`       |
| Workflow filename   | `release-crate.yml`|
| Environment         | *(leave blank)*    |

Then a tag (below) publishes; a manual run (Actions → release-crate → Run
workflow) does a `--dry-run`. The workflow also fails fast if the tag doesn't
match `Cargo.toml`'s version. To publish by hand instead: `cargo login <token>`
then `cargo publish --locked`.

docs.rs builds the documentation automatically with the `full` feature set (see
`[package.metadata.docs.rs]` in `Cargo.toml`); the `python` feature is excluded
there because it needs a Python interpreter.

## Python wheel → PyPI

The recommended path is the **`release-python.yml`** workflow, which builds
Linux (x86_64 + aarch64), macOS (x86_64 + aarch64), and Windows wheels plus an
sdist, and publishes them together via **Trusted Publishing (OIDC)** — no stored
token. One-time setup, on PyPI:

- **Existing project:** PyPI → Your projects → `millwright` → Manage →
  Publishing → Add a new publisher (GitHub Actions).
- **Before the first publish:** account → Publishing → add a *pending* publisher.

| field               | value               |
| ------------------- | ------------------- |
| Owner               | `mi7plus`           |
| Repository name     | `millwright`        |
| Workflow name       | `release-python.yml`|
| Environment         | *(leave blank)*     |

*(A pending publisher also needs the PyPI Project Name, `millwright`.)*

Then every release is just a tag:

```bash
git tag vX.Y.Z && git push --tags
```

The workflow builds all platforms and publishes on the tag. A manual run
(Actions → release-python → Run workflow) builds the wheels *without* publishing,
for a dry run.

To publish a single platform's wheel by hand instead (only installable on that
OS), from a Developer PowerShell/Prompt with a Python toolchain:

```bash
maturin publish            # needs MATURIN_PYPI_TOKEN set
```

## Notes

- The crate pins its engines to exact versions and commits `Cargo.lock`, so a
  publish is fully reproducible.
- MSRV is declared in `Cargo.toml` (`rust-version`) and enforced by cargo for
  consumers; it is dictated by transitive engine deps, so expect it to rise over
  time.
