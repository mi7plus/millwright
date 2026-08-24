# Security Policy

## Supported versions

Security fixes are provided for the latest published Millwright release. Users
should upgrade to the newest version available on crates.io or PyPI before
reporting an issue that may already have been corrected.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use
[GitHub private vulnerability reporting](https://github.com/mi7plus/millwright/security/advisories/new)
to share the affected version, impact, reproduction steps, and any suggested
mitigation privately with the maintainers.

You should receive an acknowledgement within seven days. The maintainers will
then validate the report, coordinate a fix and release where necessary, and
agree on disclosure timing with the reporter. Please allow a reasonable
remediation window before publishing details.

## Scope

Reports about the Rust crate, Python wheels, serialization and ONNX handling,
model registry, inference server, release artifacts, or build and publishing
pipeline are in scope. Vulnerabilities in an upstream dependency should also be
reported upstream when appropriate; please still notify Millwright privately if
the dependency creates a practical risk for Millwright users.
