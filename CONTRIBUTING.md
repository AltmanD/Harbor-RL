# Contributing

HarborRL accepts changes through pull requests from a public branch.

1. Fork or clone the repository and create a topic branch.
2. Install with `python -m pip install -e .[dev]`.
3. Keep the public surface native-only; do not add vendored backends, private
   datasets, deployment credentials, or machine-specific defaults.
4. Run `python -m pytest`, `python -m ruff check .`, and
   `bash scripts/audit_public_tree.sh`.
5. Add or update a CPU contract for every behavior change. Live GPU evidence is
   required before a training claim is documented as accepted.
6. Submit a reviewable commit series and describe external versions, task
   provenance, and any acceptance evidence.

Configuration and task fixtures must use generic placeholders. Absolute personal
paths, internal DNS names, private registry references, and unexplained large
files will fail review.
