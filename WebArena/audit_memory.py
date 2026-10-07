# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Snapshot memory eligibility and result artifacts without model access."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _snapshot(path: Path, data: bytes | None = None) -> dict:
    data = path.read_bytes() if data is None else data
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data)}


def _read_jsonl(path: Path, key: str) -> tuple[list[str], dict]:
    data = path.read_bytes()
    identifiers = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or key not in row:
                raise ValueError(f"missing {key!r} identifier")
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        # Match the selectors' integer/string identity convention.
        identifiers.append(str(row[key]))
    return identifiers, {**_snapshot(path, data), "row_count": len(identifiers)}


def build_manifest(task_id, memory_bank: Path, embedding_cache: Path,
                   exclude_task_ids=(), result_dir: Path | None = None,
                   model: str | None = None) -> dict:
    """Describe eligible IDs and exact input bytes; do not rank memories."""
    bank_ids, bank_snapshot = _read_jsonl(memory_bank, "task_id")
    cache_ids, cache_snapshot = _read_jsonl(embedding_cache, "id")
    current_id = str(task_id)
    excluded = {current_id, *(str(value) for value in exclude_task_ids)}
    bank_set, cache_set = set(bank_ids), set(cache_ids)
    result_artifacts, skipped = [], []
    if result_dir is not None:
        for path in sorted(result_dir.iterdir()):
            if path.is_symlink() or not path.is_file():
                skipped.append(str(path))
            else:
                result_artifacts.append(_snapshot(path))
    return {
        "schema_version": 1,
        "task_id": current_id,
        "excluded_task_ids": sorted(excluded),
        "sources": {"memory_bank": bank_snapshot, "embedding_cache": cache_snapshot},
        "isolation": {
            "eligibility_only": True,
            "eligible_candidate_ids": sorted((bank_set & cache_set) - excluded),
            "current_task_bank_rows": bank_ids.count(current_id),
            "current_task_cache_rows": cache_ids.count(current_id),
            "excluded_cache_rows": sum(value in excluded for value in cache_ids),
            "duplicate_cache_rows": len(cache_ids) - len(cache_set),
            "orphan_cache_ids": sorted(cache_set - bank_set),
        },
        "result_artifacts": result_artifacts,
        "result_directory": str(result_dir.resolve()) if result_dir is not None else None,
        "skipped_result_paths": skipped,
        "model": model,
        "cost": None,
        "cost_status": "not_recorded",
        "audit_tool": _snapshot(Path(__file__)),
    }


def write_manifest(manifest: dict, output: Path):
    """Publish atomically while preserving every audited source."""
    resolved_output = output.resolve()
    # os.replace replaces this directory entry, even when output is a symlink.
    replacement_entry = output.parent.resolve() / output.name
    sources = {Path(source["path"]) for source in manifest["sources"].values()}
    sources.add(Path(manifest["audit_tool"]["path"]))
    sources.update(Path(source["path"]) for source in manifest["result_artifacts"])
    result_dir = manifest["result_directory"]
    if resolved_output in sources or replacement_entry in sources or (result_dir is not None and
            (resolved_output.is_relative_to(result_dir) or replacement_entry.is_relative_to(result_dir))):
        raise ValueError("audit output must not overwrite a source or be inside the result directory")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, help="Use the ID stored in the memory bank, e.g., 47.")
    parser.add_argument("--memory-bank", required=True, type=Path)
    parser.add_argument("--embedding-cache", required=True, type=Path)
    parser.add_argument("--exclude-task-id", action="append", default=[])
    parser.add_argument("--result-dir", type=Path, help="Hash top-level result files; symlinks are skipped.")
    parser.add_argument("--model", help="Optional run metadata; does not select or invoke a provider.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = build_manifest(args.task_id, args.memory_bank, args.embedding_cache,
                                  args.exclude_task_id, args.result_dir, args.model)
        write_manifest(manifest, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
