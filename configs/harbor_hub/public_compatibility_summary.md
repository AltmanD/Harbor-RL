# Harbor Hub compatibility summary

HarborRL consumes the Harbor task format directly. A public Hub dataset or task can be used when it follows the Harbor task contract and passes HarborRL's content-addressed task inspection; HarborRL does not require a dataset-specific task format.

This public page reports aggregate compatibility only. It does not publish a task inventory because Hub membership, licenses, revisions, and upstream availability change over time.

## Validation summary

| Scope | Count |
| --- | ---: |
| Hub datasets scanned | 340 |
| Datasets with at least one statically compatible task | 181 |
| Statically compatible task versions | 524,041 |
| Datasets with at least one dynamically verified task | 50 |
| Dynamically verified task versions | 210 |

Counts use task content versions, not mutable Hub revision labels. Static validation checks the native Harbor task contract, image/environment declaration, instruction, verifier, and reward contract. Dynamic validation additionally builds or materializes the task and verifies reward output. A dynamic count does not imply that every task in that dataset completed RL training. Validate a newly materialized catalog with `harborrl doctor` before training.
