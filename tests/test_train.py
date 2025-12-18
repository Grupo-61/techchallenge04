"""
Teste de unidade para train.py
Realiza o importe train.py e com monkeypatch para evitar loops de treino e chamadas de rede; testa janelas_deslizantes_n_dias,
retorna_treino_teste_normalizado e o forward do SimpleLSTM (usa dummy yfinance e restaura np.arange após import).
"""

import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch


def test_nn_lstm_functions(monkeypatch):
    orig_arange = np.arange
    monkeypatch.setattr(np, "arange", lambda *args, **kwargs: [])
    import types, pandas as pd

    def _dummy_download(ticker, start=None, end=None):
        cols = pd.MultiIndex.from_tuples(
            [("Close", ""), ("Open", ""), ("High", ""), ("Low", "")]
        )
        return pd.DataFrame([[1, 2, 3, 4], [2, 3, 4, 5]], columns=cols)

    sys.modules["yfinance"] = types.SimpleNamespace(download=_dummy_download)

    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src" / "train.py"
    spec = importlib.util.spec_from_file_location("train", str(src_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(np, "arange", orig_arange)

    # Test janelas_deslizantes_n_dias
    arr = np.arange(12).reshape(3, 4)
    X, y = mod.janelas_deslizantes_n_dias(arr, seq_length=1, dias_futuros=1)
    assert X.shape[0] == len(arr) - 1

    # Test retorna_treino_teste_normalizado
    X_np = np.ones((10, 3, 1))
    y_np = np.arange(10)
    Xw = np.arange(40).reshape(10, 4)[:, :3]
    X_train, y_train, X_test, y_test, scaler = mod.retorna_treino_teste_normalizado(
        Xw, np.arange(10), train_size=6
    )
    assert hasattr(X_train, "shape") and hasattr(y_train, "shape")
    # Test SimpleLSTM forward
    model = mod.SimpleLSTM(
        input_size=1, hidden_size=8, num_layers=1, output_size=1, dropout_prob=0.1
    )
    x = torch.randn((2, 3, 1))
    out = model(x)
    assert out.shape == (2, 1)
