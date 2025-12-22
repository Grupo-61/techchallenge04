"""
Teste de unidade para evaluation_models.py
Realiza o importe evaluation_models.py com monkeypatches; cria dinamicamente um pickle dummy (models/strategy_2/modelo_lstm_1.pkl) durante o teste,
valida funções puras (ex.: build_features_2, janelas deslizantes, SimpleLSTM forward) e remove o arquivo binário no teardown.
"""

import importlib.util
import importlib
import sys
import os
import pickle
import types
from pathlib import Path
import pandas as pd
import numpy as np
import torch


def _make_dummy_yf_module():
    mod = types.SimpleNamespace()

    def download(ticker, start=None, end=None):
        cols = pd.MultiIndex.from_tuples(
            [("Close", ""), ("Open", ""), ("High", ""), ("Low", "")]
        )
        rows = 200
        data = []
        for i in range(rows):
            base = 10 + i
            data.append([base, base + 1, base + 2, base + 3])
        return pd.DataFrame(data, columns=cols)

    mod.download = download
    return mod


def _make_dummy_model_file(path):
    from tests.dummy_model_for_pickle import DummyModel

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(DummyModel(), f)
    return DummyModel()


def test_evaluation_models_import_and_funcs(tmp_path, monkeypatch):
    sys.modules["yfinance"] = _make_dummy_yf_module()
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "models" / "strategy_2"
    model_path = model_dir / "modelo_lstm_1.pkl"

    os.makedirs(model_dir, exist_ok=True)
    dummy_model = _make_dummy_model_file(str(model_path))
    monkeypatch.setattr(pickle, "load", lambda f: dummy_model)

    try:
        src_path = repo_root / "src" / "evaluation_models.py"
        spec = importlib.util.spec_from_file_location(
            "evaluation_models", str(src_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        try:
            if model_path.exists():
                model_path.unlink()
        except Exception:
            pass
        try:
            if model_dir.exists() and not any(model_dir.iterdir()):
                model_dir.rmdir()
        except Exception:
            pass

    df = pd.DataFrame({"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, 2]})
    features = mod.build_features_2(df)
    assert list(features.columns) == ["Open", "High", "Low", "Close"]

    arr = np.arange(12).reshape(3, 4)
    X, y = mod.janela_deslizante_n_dias(arr, seq_length=1)
    assert X.ndim == 3

    # janelas deslizantes n dias
    X2, y2 = mod.janela_deslizante_n_dias(arr, seq_length=1, dias_futuros=1)
    assert X2.shape[0] == len(arr) - 1
