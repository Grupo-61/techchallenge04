import io
import pickle
import yfinance as yf
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {DEVICE}")
    if torch.cuda.is_available():
        print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")


# --------------------------------------------------
# Funções de apoio
# --------------------------------------------------


def obtem_dados_historicos(ticker, data_inicial, data_final):
    dados = yf.download(ticker, start=data_inicial, end=data_final)
    colunas = []
    for col in dados.columns:
        colunas.append(col[0])
    dados.columns = colunas
    return dados


def build_features_2(data):
    columns = ["Open", "High", "Low", "Close"]
    return data[columns]


def janela_deslizante_n_dias(data, seq_length, dias_futuros=1):
    X, y = [], []
    limite = len(data) - seq_length - dias_futuros + 1
    for i in range(limite):
        X.append(data[i : i + seq_length, :])
        target_index = i + seq_length + dias_futuros - 1
        y.append(
            data[target_index, 0]
        )  # Assume que o alvo é a coluna 0 (Close, geralmente)

    return np.array(X), np.array(y)


# --------------------------------------------------
# Parâmetros de configuração
# --------------------------------------------------

    TICKER = "ITUB4.SA"
    START_DATE = "2025-06-01"
    END_DATE = "2025-10-31"
    NR_STRATEGY = 2

    # Carregando melhor modelo salvo
    model_path = f"./reports/best_strategy_{NR_STRATEGY}.csv"
    model_df = pd.read_csv(model_path)
    NR_MODEL = int(model_df.iloc[0]["nr_model"])

    # Carregando hiperparâmetros do modelo
    report = f"./reports/report_strategy_{NR_STRATEGY}.csv"
    params_df = pd.read_csv(report)
    params_df = params_df.iloc[NR_MODEL - 1]
    seq_length = int(params_df["seq_length"])
    hidden_size = int(params_df["hidden_size"])
    num_layers = int(params_df["num_layers"])
    dropout_prob = float(params_df["dropout_prob"])
    dias_futuros = int(params_df["dias_futuros"])

    # Carregando features
    features = f"./data/features/strategy_{NR_STRATEGY}.csv"
    features_df = pd.read_csv(features)
    input_size = len(features_df.columns) - 1  # Subtrai 1 para não contar a coluna Date

    print(
        f"\nModelo: {NR_MODEL} | seq_length: {seq_length} | hidden_size: {hidden_size} | num_layers: {num_layers} | dropout_prob: {dropout_prob}"
    )

    # --------------------------------------------------
    # Coletando dados de validação
    # --------------------------------------------------

    print("\nObtendo dados de validação...")

    data = obtem_dados_historicos(TICKER, START_DATE, END_DATE)
    data = build_features_2(data)
    data = data.to_numpy()

    X, y = janela_deslizante_n_dias(data, seq_length, dias_futuros)
    X_test_reshaped = X.reshape(-1, 1)
    y_test_reshaped = y.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_test_reshaped)

    X_test_norm = scaler.transform(X_test_reshaped).reshape(X.shape)
    y_test_norm = scaler.transform(y.reshape(-1, 1))
    X_test = torch.from_numpy(X_test_norm).float().to(DEVICE).unsqueeze(-1)
    y_test = torch.from_numpy(y_test_norm).float().to(DEVICE).unsqueeze(1)
    
    # --------------------------------------------------
    # Carregando modelo treinado
    # --------------------------------------------------
    
    print("\nCarregando modelo treinado...")
    
    arquivo_modelo = f"./models/strategy_2/modelo_lstm_{NR_MODEL}.pkl"
    
    # Caso GPU disponível
    if torch.cuda.is_available():
        try:
            with open(arquivo_modelo, "rb") as f:
                model = pickle.load(f)
    
            print("Modelo carregado com sucesso na GPU.")
            # Se for um modelo PyTorch, lembre-se de colocá-lo no modo de avaliação
            if hasattr(model, "eval"):
                model.eval()
    
        except Exception as e:
            print(f"Erro ao carregar modelo na GPU: {e}")
    
    # Caso GPU não disponível - Forçar carregamento na CPU
    else:
    
        class CpuUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch.storage" and name == "_load_from_bytes":
                    # Retorna uma função lambda que chama torch.load forçando a CPU
                    return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
                else:
                    # Para o resto, usa o comportamento padrão
                    return super().find_class(module, name)
    
        try:
            with open(arquivo_modelo, "rb") as f:
                # Usa o Unpickler personalizado para carregar o objeto
                model = CpuUnpickler(f).load()
    
            print("Modelo carregado com sucesso na CPU usando CPU_Unpickler.")
            # Se for um modelo PyTorch, lembre-se de colocá-lo no modo de avaliação
            if hasattr(model, "eval"):
                model.eval()
    
        except Exception as e:
            print(f"Erro ao carregar com CPU_Unpickler: {e}")
    
    # --------------------------------------------------
    # Imprimindo gráfico de desempenho e métricas de erro
    # --------------------------------------------------
    
    print("Plotando gráfico para acompanhar desempenho...")
    
    model.eval()
    with torch.no_grad():
        test_predictions_norm = model(X_test.squeeze(3)).cpu().numpy()
    
        y_test_pronto = y_test.view(-1, 1)
        test_predictions = scaler.inverse_transform(test_predictions_norm)
        actual_prices = scaler.inverse_transform(y_test_pronto.cpu().numpy())
    
    saida_plot = f"./reports/plots/graph_strategy_2_lstm.png"
    
    plt.figure(figsize=(16, 9))
    plt.plot(
        pd.DataFrame(actual_prices).index[:],
        pd.DataFrame(actual_prices),
        label="Preço Real",
        color="blue",
    )
    plt.plot(
        pd.DataFrame(test_predictions).index[:],
        pd.DataFrame(test_predictions),
        label="Previsão LSTM",
        color="red",
    )
    plt.title(f"Previsão de Preço de Fechamento do {TICKER} (LSTM)")
    plt.xlabel("Data")
    plt.ylabel("Preço (R$)")
    plt.legend()
    plt.grid(True)
    plt.savefig(saida_plot, bbox_inches="tight")
    plt.close()
    
    
    print("\nAvaliando métricas de erro..")
    
    
    mae = mean_absolute_error(actual_prices, test_predictions)
    print(f"\nMAE no Conjunto de Validação: R$ {mae:.4f}")
    
    
    mse = mean_squared_error(actual_prices, test_predictions)
    print(f"MSE no Conjunto de Validação: R$ {mse:.4f}")
    
    
    rmse = root_mean_squared_error(actual_prices, test_predictions)
    print(f"RMSE no Conjunto de Validação: R$ {rmse:.4f}")
    
    
    dac = np.mean(
        (
            np.sign(actual_prices[1:] - actual_prices[:-1])
            == np.sign(test_predictions[1:] - test_predictions[:-1])
        ).astype(int)
    )
    print(f"DAC no Conjunto de Validação: {dac:.2%}")
