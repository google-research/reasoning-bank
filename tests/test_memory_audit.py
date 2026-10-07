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

"""Offline artifact audit tests; no SDKs, credentials or model calls."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "WebArena/audit_memory.py"


def load_audit():
    spec = importlib.util.spec_from_file_location("memory_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemoryAuditTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.bank = self.root / "bank.jsonl"
        self.cache = self.root / "cache.jsonl"
        self.bank.write_text(''.join(json.dumps({"task_id": tid, "memory_items": ["synthetic"]}) + '\n'
                                     for tid in (21, "22", 23)))
        self.cache.write_text(''.join(json.dumps({"id": tid, "embedding": [1, 0]}) + '\n'
                                      for tid in ("21", 22, "22", "23", "orphan")))
        self.audit = load_audit()

    def test_self_explicit_exclusions_duplicates_and_orphans(self):
        report = self.audit.build_manifest(21, self.bank, self.cache, exclude_task_ids=[23])
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["task_id"], "21")
        self.assertEqual(report["excluded_task_ids"], ["21", "23"])
        isolation = report["isolation"]
        self.assertEqual(isolation["eligible_candidate_ids"], ["22"])
        self.assertEqual(isolation["current_task_cache_rows"], 1)
        self.assertEqual(isolation["excluded_cache_rows"], 2)
        self.assertEqual(isolation["duplicate_cache_rows"], 1)
        self.assertEqual(isolation["orphan_cache_ids"], ["orphan"])
        self.assertTrue(isolation["eligibility_only"])

    def test_input_hashes_cover_exact_bytes_and_inputs_are_unchanged(self):
        before = {path: path.read_bytes() for path in (self.bank, self.cache)}
        report = self.audit.build_manifest("21", self.bank, self.cache)
        for role, path in (("memory_bank", self.bank), ("embedding_cache", self.cache)):
            self.assertEqual(path.read_bytes(), before[path])
            self.assertEqual(report["sources"][role]["sha256"], hashlib.sha256(before[path]).hexdigest())
            self.assertEqual(report["sources"][role]["size_bytes"], len(before[path]))
        self.assertEqual(report["sources"]["embedding_cache"]["row_count"], 5)
        self.assertIsNone(report["cost"])

    def test_empty_sources_have_no_eligible_candidates(self):
        self.bank.write_text('\n')
        self.cache.write_text('')
        report = self.audit.build_manifest("new", self.bank, self.cache)
        self.assertEqual(report["isolation"]["eligible_candidate_ids"], [])
        self.assertEqual(report["sources"]["memory_bank"]["row_count"], 0)
        self.assertEqual(report["sources"]["embedding_cache"]["row_count"], 0)

    def test_result_snapshot_includes_unknown_evaluation_without_inferring_cost(self):
        results = self.root / "webarena.21"
        results.mkdir()
        artifact = results / "model_autoeval.json"
        artifact.write_text('[{"uid":21,"rm":null}]')
        (results / "summary_info.json").write_text('{"cum_reward":1}')
        (results / "linked.json").symlink_to(artifact)
        report = self.audit.build_manifest("21", self.bank, self.cache, result_dir=results, model="fixture")
        self.assertEqual(report["model"], "fixture")
        self.assertIsNone(report["cost"])
        snapshot = next(row for row in report["result_artifacts"] if row["path"] == str(artifact.resolve()))
        self.assertEqual(snapshot["sha256"], hashlib.sha256(artifact.read_bytes()).hexdigest())
        self.assertEqual(report["skipped_result_paths"], [str(results / "linked.json")])

    def test_invalid_jsonl_has_source_and_line_context(self):
        for content in ('\n{bad json}\n', '\n{"text":"missing identifier"}\n'):
            with self.subTest(content=content):
                self.cache.write_text(content)
                with self.assertRaisesRegex(ValueError, "cache.jsonl:2"):
                    self.audit.build_manifest("21", self.bank, self.cache)

    def command(self, output):
        return [sys.executable, str(SCRIPT), '--task-id', '21', '--memory-bank', str(self.bank),
                '--embedding-cache', str(self.cache), '--exclude-task-id', '23', '--output', str(output)]

    def test_real_cli_writes_manifest_without_provider_dependencies(self):
        output = self.root / "audit.json"
        run = subprocess.run(self.command(output), capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report["isolation"]["eligible_candidate_ids"], ["22"])
        self.assertEqual(report["excluded_task_ids"], ["21", "23"])
        self.assertIsNone(report["cost"])

    def test_output_cannot_overwrite_input(self):
        before = self.bank.read_bytes()
        run = subprocess.run(self.command(self.bank), capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("overwrite", run.stderr)
        self.assertEqual(self.bank.read_bytes(), before)

    def test_output_cannot_overwrite_result_artifact(self):
        results = self.root / "results"
        results.mkdir()
        artifact = results / "summary_info.json"
        artifact.write_text('{"cum_reward":1}')
        before = artifact.read_bytes()
        run = subprocess.run(self.command(artifact) + ['--result-dir', str(results)],
                             capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(artifact.read_bytes(), before)

    def test_output_symlink_entry_inside_results_is_preserved(self):
        results = self.root / "results"
        results.mkdir()
        external = self.root / "external.json"
        external.write_text('external data')
        output = results / "audit.json"
        output.symlink_to(external)
        run = subprocess.run(self.command(output) + ['--result-dir', str(results)],
                             capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertTrue(output.is_symlink())
        self.assertEqual(external.read_text(), 'external data')

    def test_helper_protects_result_directory_from_manifest(self):
        results = self.root / "results"
        results.mkdir()
        artifact = results / "summary_info.json"
        artifact.write_text('{"cum_reward":1}')
        before = artifact.read_bytes()
        manifest = self.audit.build_manifest("21", self.bank, self.cache, result_dir=results)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.audit.write_manifest(manifest, artifact)
        self.assertEqual(artifact.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "result directory"):
            self.audit.write_manifest(manifest, results / "new-audit.json")
        self.assertFalse((results / "new-audit.json").exists())


if __name__ == '__main__':
    unittest.main()
