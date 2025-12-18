'''
 Configura mocks globais usados por todos os testes: um torch leve (inclui nn, zeros, no_grad, utils.data minimal),
 sklearn.preprocessing.MinMaxScaler e sklearn.metrics simples, 
 matplotlib.pyplot noop e um yfinance dummy. Isso permite rodar a suíte de testes sem instalar torch, scikit-learn, matplotlib ou yfinance.
'''

import sys
import types
import numpy as np
from pathlib import Path

class FakeTensor:
    def __init__(self, array):
        self._arr = np.array(array)

    def size(self, dim=None):
        if dim is None:
            return self._arr.shape
        return self._arr.shape[dim]

    @property
    def shape(self):
        return self._arr.shape

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def float(self):
        return self

    def view(self, *shape):
        return FakeTensor(self._arr.reshape(shape))

    def squeeze(self, dim=None):
        if dim is None:
            return FakeTensor(self._arr.squeeze())
        return FakeTensor(np.squeeze(self._arr, axis=dim))

    def unsqueeze(self, dim):
        arr = np.expand_dims(self._arr, axis=dim)
        return FakeTensor(arr)

    def __getitem__(self, idx):
        return FakeTensor(self._arr[idx])

    def __array__(self):
        return self._arr

    def to(self, device):
        return self

class FakeNN:
    class Module:
        def __call__(self, *args, **kwargs):
            if hasattr(self, 'forward'):
                return self.forward(*args, **kwargs)
            raise TypeError('Module has no forward method')

    class LSTM:
        def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
            self.hidden_size = hidden_size

        def __call__(self, x, states=None):
            # x is FakeTensor with shape (batch, seq_length, input_size)
            b, s, _ = x.shape
            out = FakeTensor(np.zeros((b, s, self.hidden_size)))
            return out, None

    class Dropout:
        def __init__(self, p):
            pass

    class Linear:
        def __init__(self, in_features, out_features):
            self.out_features = out_features

        def __call__(self, x):
            # x: FakeTensor with shape (batch, in_features)
            b = x.shape[0]
            return FakeTensor(np.zeros((b, self.out_features)))

    class MSELoss:
        def __call__(self, a, b):
            return 0.0

    class L1Loss:
        def __call__(self, a, b):
            return 0.0


class FakeTorch(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.nn = FakeNN
        self.cuda = types.SimpleNamespace(is_available=lambda: False, get_device_name=lambda i: 'cpu')
        self.device = lambda x: x

    def no_grad(self):
        class _NoGrad:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc, tb):
                return False
        return _NoGrad()
    def zeros(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, np.ndarray)):
            shape = tuple(shape[0])
        return FakeTensor(np.zeros(shape))

    def from_numpy(self, arr):
        return FakeTensor(np.array(arr))

    def randn(self, shape):
        return FakeTensor(np.random.randn(*shape))

    def __getattr__(self, name):
        # Provide basic numpy wrappers for common ops if requested
        if name == 'tensor':
            return lambda arr: FakeTensor(np.array(arr))
        raise AttributeError(name)


def pytest_sessionstart(session):
    # Insert fake torch module into sys.modules before tests import anything
    fake_torch = FakeTorch()
    sys.modules['torch'] = fake_torch
    sys.modules['torch.nn'] = fake_torch.nn
    sys.modules['torch.cuda'] = fake_torch.cuda
    # Provide minimal torch.utils and torch.utils.data modules
    import types as _types
    torch_utils = _types.ModuleType('torch.utils')
    torch_utils_data = _types.ModuleType('torch.utils.data')

    class Dataset:
        pass

    class DataLoader:
        def __init__(self, dataset, batch_size=1, shuffle=False):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle

        def __iter__(self):
            return iter([])

    torch_utils_data.Dataset = Dataset
    torch_utils_data.DataLoader = DataLoader
    sys.modules['torch.utils'] = torch_utils
    sys.modules['torch.utils.data'] = torch_utils_data
    # Minimal sklearn.preprocessing.MinMaxScaler to avoid installing scikit-learn
    sklearn_module = __import__('types')
    _mod = sklearn_module.ModuleType('sklearn')
    _prep = sklearn_module.ModuleType('sklearn.preprocessing')

    class MinMaxScaler:
        def __init__(self, feature_range=(-1, 1)):
            self.feature_range = feature_range
            self.data_min_ = None
            self.data_max_ = None

        def fit(self, X):
            arr = np.array(X)
            self.data_min_ = arr.min(axis=0)
            self.data_max_ = arr.max(axis=0)
            return self

        def transform(self, X):
            arr = np.array(X)
            data_min = self.data_min_
            data_max = self.data_max_
            scale = (self.feature_range[1] - self.feature_range[0]) / (data_max - data_min + 1e-9)
            return (arr - data_min) * scale + self.feature_range[0]

        def fit_transform(self, X):
            return self.fit(X).transform(X)

        def inverse_transform(self, X):
            arr = np.array(X)
            data_min = self.data_min_
            data_max = self.data_max_
            scale = (self.feature_range[1] - self.feature_range[0]) / (data_max - data_min + 1e-9)
            return (arr - self.feature_range[0]) / (scale + 1e-9) + data_min

    _prep.MinMaxScaler = MinMaxScaler
    sys.modules['sklearn'] = _mod
    sys.modules['sklearn.preprocessing'] = _prep
    # Minimal sklearn.metrics
    metrics_mod = sklearn_module.ModuleType('sklearn.metrics')

    def mean_absolute_error(a, b):
        return float(np.mean(np.abs(np.array(a) - np.array(b))))

    def mean_squared_error(a, b):
        return float(np.mean((np.array(a) - np.array(b)) ** 2))

    def root_mean_squared_error(a, b):
        return float(np.sqrt(mean_squared_error(a, b)))

    metrics_mod.mean_absolute_error = mean_absolute_error
    metrics_mod.mean_squared_error = mean_squared_error
    metrics_mod.root_mean_squared_error = root_mean_squared_error
    sys.modules['sklearn.metrics'] = metrics_mod
    # Ensure src/ is on sys.path so `import strategy` and similar work
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    # Minimal matplotlib.pyplot mock
    import types as _types2
    mpl = _types2.ModuleType('matplotlib')
    mpl_pyplot = _types2.ModuleType('matplotlib.pyplot')

    def _noop(*args, **kwargs):
        return None

    mpl_pyplot.figure = _noop
    mpl_pyplot.plot = _noop
    mpl_pyplot.title = _noop
    mpl_pyplot.xlabel = _noop
    mpl_pyplot.ylabel = _noop
    mpl_pyplot.legend = _noop
    mpl_pyplot.grid = _noop
    mpl_pyplot.savefig = _noop
    mpl_pyplot.close = _noop

    sys.modules['matplotlib'] = mpl
    sys.modules['matplotlib.pyplot'] = mpl_pyplot


# Also provide a fixture to ensure yfinance is mocked (some modules import it at top-level)
import pandas as pd
import types as _types

class _DummyYF(_types.SimpleNamespace):
    def download(self, ticker, start=None, end=None):
        cols = pd.MultiIndex.from_tuples([('Close', ''), ('Open', ''), ('High', ''), ('Low', '')])
        return pd.DataFrame([[10,11,12,13],[20,21,22,23]], columns=cols)

sys.modules['yfinance'] = _DummyYF()
