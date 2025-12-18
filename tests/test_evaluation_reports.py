'''
Teste verifica se o loop top-level (monkeypatch em np.arange)
 para garantir que o módulo importa sem executar processamento de arquivos;
 valida que a importação ocorre sem erros.
'''

import importlib.util
import sys
from pathlib import Path

import numpy as np


def test_evaluation_reports_import(monkeypatch):
    monkeypatch.setattr(np, 'arange', lambda *args, **kwargs: [])

    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / 'src' / 'evaluation_reports.py'
    spec = importlib.util.spec_from_file_location('evaluation_reports', str(src_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert 'evaluation_reports' in sys.modules or mod is not None
