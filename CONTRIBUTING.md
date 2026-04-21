## Contributing

Thank you for contributing to Reflexive Radio Adapter.

## Development Setup

1. Create and activate a Python virtual environment.
2. Install dependencies from requirements.txt.
3. Run training/evaluation scripts with explicit CLI arguments.

## Pull Request Guidelines

1. Keep pull requests focused and small.
2. Add clear descriptions of what changed and why.
3. Include reproduction commands used for validation.
4. Do not include protected clinical data or PHI.
5. Do not commit large checkpoint binaries unless the PR explicitly targets model artifact release.

## Code Quality Expectations

1. Preserve reproducibility by documenting seeds and hyperparameters.
2. Avoid hardcoded machine-specific paths.
3. Prefer CLI/config driven settings.
4. Keep logging clear and structured.

## Reporting Issues

Please include:
- Environment details (OS, Python, CUDA, GPU)
- Exact command used
- Full traceback or log snippet
- Steps to reproduce
