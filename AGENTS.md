## Lesson Learned: Cache Policy

Do not treat reproducibility as a default requirement in this repo.

For normal script runs, the goal is internal consistency within the current run, not automatic invalidation of every cached model result after code or parameter changes.

Only enforce full reproducibility and full cache-refresh behavior in the canonical end-to-end `run_all.py` workflow, where a complete rerun is the point.

## Lesson Learned: Python Environment

Use the project-local virtual environment for repo commands.

- Prefer `.venv/bin/python` over `python` or `python3` when running repository scripts.
- Prefer `.venv/bin/pip` for package inspection inside this repo.
- Assume system Python may be missing required packages even when `.venv` is correctly provisioned.
