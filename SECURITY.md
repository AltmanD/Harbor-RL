# Security policy

## Supported release

Security fixes target the latest public `v0.x` release line.

## Reporting

Do not open a public issue for a suspected vulnerability. Contact the
maintainers through the repository's private GitHub security advisory interface.
Include reproduction steps, affected versions, and logs with credentials removed.

## Boundaries

HarborRL executes untrusted task instructions and verifiers in external Harbor
Docker workers. Deploy it only on isolated workers with bounded CPU, memory,
storage, and network access. Provider credentials belong only on the controlled
training/gateway side; workers receive per-attempt gateway credentials. The
framework rejects missing serving evidence, stale policy versions, altered task
digests, and reward artifacts that disagree with verifier results, but it cannot
replace container isolation or host hardening.
