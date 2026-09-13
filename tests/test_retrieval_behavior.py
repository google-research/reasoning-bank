"""Execute real retrieval functions with deterministic numeric/provider doubles.

AST extraction avoids module-level cloud clients; function bodies are unmodified.
No source-text assertions, credentials, downloads or model calls are involved.
"""
import ast
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PATHS = ('WebArena/memory_management.py',
         'third_party/src/minisweagent/memory/memory_management.py')


class Tensor:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    @property
    def T(self):
        return Tensor(list(map(list, zip(*self.data))))

    def __matmul__(self, other):
        return Tensor([[sum(a*b for a, b in zip(row, col))
                        for col in zip(*other.data)] for row in self.data])

    def squeeze(self, axis):
        return Tensor(self.data[0]) if len(self.data) == 1 else self

    def __mul__(self, factor):
        return Tensor([v * factor for v in self.data])

    def tolist(self):
        return self.data


class Torch:
    float32 = None
    tensor = staticmethod(lambda data, **kw: Tensor(data))
    empty = staticmethod(lambda n: Tensor([]))


def load_functions(path):
    names = {'load_cached_embeddings', 'get_detailed_instruct', 'select_memory', 'screening'}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    import os
    env = dict(json=json, os=os, torch=Torch, logger=logging.getLogger('retrieval-test'),
               l2_normalize=lambda value, **kw: value,
               embed_query_with_gemini=lambda *a, **kw: Tensor([[1., 0.]]),
               embed_query_with_qwen=lambda *a, **kw: Tensor([[1., 0.]]))
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), env)
    return env


class RetrievalBehavior(unittest.TestCase):
    def test_canary_legacy_signature(self):
        def check(env, cache, bank):
            result = env['select_memory'](1, bank, 'q', task_id='t1', cache_path=str(cache))
            self.assertEqual(result, [bank[1]])
        self.run_case(check)

    def run_case(self, check):
        for relative in PATHS:
            with self.subTest(path=relative), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                env = load_functions(ROOT / relative)
                cache = Path(directory) / 'cache.jsonl'
                rows = [{'id': 't1', 'text': 'self', 'embedding': [1., 0.]},
                        {'id': 't2', 'text': 'other', 'embedding': [.8, .6]},
                        {'id': 't3', 'text': 'tie', 'embedding': [.8, .6]}]
                cache.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                bank = [{'task_id': row['id'], 'memory_items': [row['text']]} for row in rows]
                check(env, cache, bank)

    def test_excludes_self_before_top_n_and_preserves_ties(self):
        def check(env, cache, bank):
            scores, ids = env['screening']('q', str(cache), task_id='t1', append_current=False)
            self.assertEqual(ids, ['t2', 't3'])
            self.assertEqual([sid for sid, _ in scores], ids)
            result = env['select_memory'](1, bank, 'q', task_id='t1', cache_path=str(cache), append_current=False)
            self.assertEqual(result, [bank[1]])
        self.run_case(check)

    def test_explicit_exclusions_union_current_id(self):
        def check(env, cache, bank):
            result = env['select_memory'](2, bank, 'q', task_id='t1', cache_path=str(cache), exclude_task_ids=['t2'], append_current=False)
            self.assertEqual(result, [bank[2]])
        self.run_case(check)

    def test_all_candidates_excluded(self):
        def check(env, cache, bank):
            result = env['select_memory'](2, bank, 'q', task_id='t1', cache_path=str(cache), exclude_task_ids=['t2','t3'], append_current=False)
            self.assertFalse(result)
        self.run_case(check)

    def test_read_only_cache_bytes_unchanged(self):
        def check(env, cache, bank):
            before = cache.read_bytes()
            env['screening']('q', str(cache), task_id='t1', append_current=False)
            self.assertEqual(cache.read_bytes(), before)
        self.run_case(check)

    def test_read_only_missing_cache_not_created(self):
        def check(env, cache, bank):
            missing = cache.parent / 'missing.jsonl'
            self.assertEqual(env['screening']('q', str(missing), task_id='t1', append_current=False), ([], []))
            self.assertFalse(missing.exists())
        self.run_case(check)

    def test_default_append_keeps_runner_ingestion_working(self):
        def check(env, cache, bank):
            env['select_memory'](1, bank, 'q', task_id='new', cache_path=str(cache))
            self.assertEqual(len(cache.read_text().splitlines()), 4)
            self.assertEqual(json.loads(cache.read_text().splitlines()[-1])['id'], 'new')
        self.run_case(check)

    def test_integer_and_string_ids_map(self):
        def check(env, cache, bank):
            cache.write_text(json.dumps({'id': 2, 'text': 'other', 'embedding': [1., 0.]})+'\n')
            self.assertEqual(env['select_memory'](1, [{'task_id': 2}], 'q', task_id=1, cache_path=str(cache), append_current=False), [{'task_id': 2}])
            self.assertFalse(env['select_memory'](1, [{'task_id': 2}], 'q', task_id='2', cache_path=str(cache), append_current=False))
        self.run_case(check)

    def test_mapping_defends_against_stale_self_and_orphan_ids(self):
        def check(env, cache, bank):
            with patch.dict(env, screening=lambda **kw: ([], ['t1', 'orphan', 't2', 't2', 't3'])):
                result = env['select_memory'](2, bank, 'q', task_id='t1', cache_path=str(cache))
            self.assertEqual(result, bank[1:])
        self.run_case(check)


if __name__ == '__main__':
    unittest.main()
