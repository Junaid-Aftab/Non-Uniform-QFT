import sys
from pathlib import Path

def find_repo_root(start: Path | None = None) -> Path:
    start = start or Path.cwd()
    for p in [start, *start.parents]:
        if (p / "src").is_dir() and ((p / "pyproject.toml").exists() or (p / ".git").exists()):
            return p
    raise RuntimeError(f"Could not find repo root from {start}")

repo_root = find_repo_root()
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print("Repo root:", repo_root)
print("Added to sys.path:", src_path)