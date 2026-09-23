# Release scope

## Included in public v0.1

- Strict native schema-2 configuration and CLI.
- Messages gateway with serving-evidence enforcement.
- External Harbor job runner and Native IR v2 validation.
- Backend-neutral immutable training batch export.
- Official Slime v0.3.2 adapter with pinned version gates.
- CPU contracts, offline hello-world example, Qwen dry-run YAML, CI, package
  build, and public-tree hygiene audit.

## Explicitly excluded

- Vendored Slime/Megatron backends.
- Private benchmarks, datasets, run artifacts, deployment manifests, and worker
  operations.
- Legacy interactive exporters, research algorithms, evaluation tools, and
  non-native harnesses.
- Model weights, container images, credentials, or internal host defaults.

## Blocking acceptance

- [x] Release branch is based on `refactor/slime-v032-exporter-dev`.
- [x] Public package contains only the native framework and v0.3.2 adapter.
- [x] Five core CPU suites pass.
- [x] Offline example, schema-2 dry-run, syntax checks, and hygiene audit pass.
- [x] Package build plan is covered by CI.
- [ ] Harbor Hub compatibility matrix passes.
- [ ] Pinned GPU one-step acceptance passes and evidence is recorded.
- [ ] Clean-clone install and wheel smoke pass in CI on the release commit.
- [ ] Maintainer review and merge to `main`; tag only after merge.

Historical repositories are not rewritten. Known risks already accepted by the
owner remain in old unreachable-by-release history, but no new secret, private
host, private data, or personal path may enter this branch's diff.
