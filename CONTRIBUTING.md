# Contributing

Thanks for helping. `main` is protected: **all changes go through a pull request.** Direct pushes and force-pushes to `main`, and deleting or moving version tags (`v*`), are blocked.

## Workflow

1. **Fork** the repo (outside contributors) or create a branch (collaborators): `git checkout -b feat/<short-name>`.
2. Make the change with tests. Run `pytest -q` — all tests must pass.
3. If the change affects a model or its results, add the new measurements to `results/` (one commit per step) and **do not overwrite an existing version's results or weights**; propose a new version instead (see [docs/versions/README.md](docs/versions/README.md)).
4. Open a pull request against `main` and fill in the template. The code owner is requested for review automatically.
5. Merge after review and green tests. Squash or merge commits are both fine; keep messages descriptive.

## Rules

- Never commit secrets, credentials, device passwords, or datasets (`data/`), weights (`*.pt`), or PDFs.
- Keep metrics honest: same held-out splits for before/after, and say what a number measures.
- Commit messages: `feat:`, `fix:`, `docs:`, `results:`, `chore:` prefixes.

## Local setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
pytest -q
```
