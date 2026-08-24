# Releasing

Millwright ships as one Rust crate (crates.io) and one Python wheel (PyPI), from
the same source.

## The easy way

With `main` green and `CHANGELOG.md`'s `[Unreleased]` section filled in, run:

```bash
# Linux/macOS
bash scripts/release.sh 2.2.1

# Windows PowerShell
powershell -File scripts/release.ps1 2.2.1
```

It synchronizes both manifests and every public HTML version marker, rolls the
changelog, syncs `Cargo.lock`, verifies the package, commits, tags `v2.2.1`, and
(after a prompt) pushes — which triggers the
single gated release workflow below. The rest of this file is the manual
equivalent.

## Preflight (manual)

1. `main` is green in CI and `CHANGELOG.md`'s `[Unreleased]` section is current.
2. Run `python scripts/sync-version.py X.Y.Z`, move the changelog's
   `[Unreleased]` entries under the new version, and commit.
3. Prove the packaged crate builds exactly as crates.io will build it:

   ```bash
   cargo publish --dry-run --locked
   ```

## Coordinated crates.io and PyPI release

Automated by **`release.yml`**, which publishes on a `v*` tag via **Trusted
Publishing (OIDC)** — no token is stored anywhere. It first verifies that the
tagged commit is on `main`, has a successful CI run, has synchronized version
markers, and has a dated changelog section. It then dry-runs the crate package,
builds every wheel and the sdist with pinned Rust and Python versions, and
installs and tests every native wheel.
Neither registry is touched until every gate passes.

If the tag arrives while CI is still running, release preflight waits for it for
up to 30 minutes. Publication is safe to rerun after a registry interruption:
an existing crate is skipped, while PyPI uploads only files that are still
missing. After both registries succeed, the workflow creates the GitHub Release
and attaches all wheels, the source distribution, the `.crate` archive, an SPDX
SBOM generated from that staged artifact bundle, and a SHA-256 manifest. GitHub
records signed provenance and artifact-bundle SBOM attestations for every
listed package.

Two registries cannot provide a cross-registry transaction: an external outage
during the final publish job can still leave one registry ahead of the other.
Running both publishes in one job after all validation makes that unavoidable
window as small and recoverable as possible.

Configure crates.io (crate → Settings → Trusted Publishing → add GitHub):

> Existing trusted-publisher entries for the retired `release-crate.yml` and
> `release-python.yml` workflows must be replaced with `release.yml` before the
> next version tag is pushed.

| field               | value              |
| ------------------- | ------------------ |
| Repository owner    | `mi7plus`          |
| Repository name     | `millwright`       |
| Workflow filename   | `release.yml`      |
| Environment         | *(leave blank)*    |

To publish by hand instead: `cargo login <token>` then `cargo publish --locked`.

docs.rs builds the documentation automatically with the `full` feature set (see
`[package.metadata.docs.rs]` in `Cargo.toml`); the `python` feature is excluded
there because it needs a Python interpreter.

Configure PyPI:

- **Existing project:** PyPI → Your projects → `millwright` → Manage →
  Publishing → Add a new publisher (GitHub Actions).
- **Before the first publish:** account → Publishing → add a *pending* publisher.

| field               | value               |
| ------------------- | ------------------- |
| Owner               | `mi7plus`           |
| Repository name     | `millwright`        |
| Workflow name       | `release.yml`       |
| Environment         | *(leave blank)*     |

*(A pending publisher also needs the PyPI Project Name, `millwright`.)*

Then every release is just a tag:

```bash
git tag vX.Y.Z && git push --tags
```

The workflow builds Linux (x86_64 + aarch64), macOS (x86_64 + aarch64), and
Windows x64 wheels plus an sdist. A manual run (Actions → release → Run
workflow) performs every build, test, SBOM, provenance-attestation, and
attestation-verification gate *without* publishing.

For the first tagged run after changing publication automation, verify the
tag-only registry and GitHub Release steps in the successful workflow log. The
artifact and attestation path is exercised by every manual run. Independently
verify a downloaded artifact with:

```bash
gh attestation verify path/to/artifact --repo mi7plus/millwright
```

To publish a single platform's wheel by hand instead (only installable on that
OS), from a Developer PowerShell/Prompt with a Python toolchain:

```bash
maturin publish            # needs MATURIN_PYPI_TOKEN set
```

## Notes

- Release tags matching `v*` are immutable: the repository ruleset permits
  creation but blocks later updates and deletion.
- GitHub Releases are immutable too: after publication, their assets and release
  metadata cannot be replaced. Corrections require a new version.
- The crate pins its engines to exact versions, commits `Cargo.lock`, and pins
  release Rust/Python versions. GitHub-hosted runner image revisions can still
  change beneath their dated labels, so releases are dependency-resolved and
  tightly controlled rather than guaranteed byte-for-byte reproducible.
- MSRV is declared in `Cargo.toml` (`rust-version`) and enforced by cargo for
  consumers; it is dictated by transitive engine deps, so expect it to rise over
  time.
