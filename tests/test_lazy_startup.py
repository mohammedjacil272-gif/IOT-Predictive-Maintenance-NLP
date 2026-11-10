from __future__ import annotations

import builtins
import os
import shutil
import subprocess
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


def test_fastapi_import_does_not_import_torch(tmp_path):
    script = """
import builtins
import os
import sys

os.environ['IOT_INTEGRATED_ROOT'] = r'{root}'
original_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('torch imported during FastAPI startup')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked
import app.main
assert 'torch' not in sys.modules
print('startup-ok')
""".format(root=str(tmp_path))
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'startup-ok'


def test_rule_based_nlp_does_not_require_torch(tmp_path):
    script = """
import builtins
import os
import sys

os.environ['IOT_INTEGRATED_ROOT'] = r'{root}'
original_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('torch imported by rule-based NLP')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked
from nlp.ner import extract_entities
from nlp.severity import score_severity
assert extract_entities('Critical bearing vibration')
assert score_severity('Critical bearing vibration')['severity_label'] == 'critical'
assert 'torch' not in sys.modules
print('rule-based-ok')
""".format(root=str(tmp_path))
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'rule-based-ok'


def test_missing_optional_torch_is_feature_specific(monkeypatch):
    from nlp.embeddings import TextEmbedder

    fake_transformers = types.ModuleType('transformers')
    fake_transformers.AutoTokenizer = type('AutoTokenizer', (), {'from_pretrained': staticmethod(lambda *a, **k: object())})
    fake_transformers.AutoModel = type('AutoModel', (), {'from_pretrained': staticmethod(lambda *a, **k: object())})
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.'):
            raise ImportError('torch missing in fixture')
        return original_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    monkeypatch.setattr(builtins, '__import__', blocked)
    with pytest.raises(RuntimeError, match="optional 'torch' is not installed"):
        TextEmbedder()


def test_model_loading_is_lazy_cached_and_thread_safe(tmp_path, monkeypatch):
    import torch

    from iot import lstm_inference

    source = Path('artifacts/models/fd001_lstm_baseline/best_model.pt')
    checkpoint = tmp_path / 'fixture-model.pt'
    shutil.copyfile(source, checkpoint)
    calls = 0
    calls_lock = threading.Lock()
    original_load = torch.load

    def counted_load(*args, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', counted_load)
    with ThreadPoolExecutor(max_workers=4) as pool:
        models = list(pool.map(lambda _: lstm_inference.load_frozen_model(checkpoint), range(4)))
    assert calls == 1
    assert all(model is models[0] for model in models)
