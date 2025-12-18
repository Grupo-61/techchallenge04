# Instalação do PyTorch com suporte CUDA (GPU) NVIDIA MX150, CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

import os
import time
import pickle
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import SimpleLSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from strategy import estrategia_negociacao_1
from strategy import estrategia_negociacao_2
from strategy import estrategia_negociacao_3
from strategy import estrategia_negociacao_4
from strategy import estrategia_negociacao_5
from strategy import estrategia_negociacao_6

# Verifica disponibilidade da GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Dispositivo: {str(DEVICE).upper()}, GPU: {torch.cuda.get_device_name(0)}\n")


# Download da série histórica do ativo
def obtem_dados_historicos(ticker, data_inicial, data_final):
    dados = yf.download(ticker, start=data_inicial, end=data_final)
    colunas = []
    for col in dados.columns:
        colunas.append(col[0])

    dados.columns = colunas
    return dados


# Controi a janela deslizante
def janelas_deslizantes_n_dias(data, seq_length, dias_futuros=1):
    X, y = [], []
    limite = len(data) - seq_length - dias_futuros + 1
    for i in range(limite):
        X.append(data[i : i + seq_length, :])
        target_index = i + seq_length + dias_futuros - 1

        # Assume que o alvo é a coluna 0 (Close, geralmente)
        y.append(data[target_index, 0])

    return np.array(X), np.array(y)


# Retorna dados treino e teste tratados/normalizados
def retorna_treino_teste_normalizado(X_np, y_np, train_size):

    # Divide as sequências em Treino e Teste
    X_train_np_raw, X_test_np_raw = X_np[:train_size], X_np[train_size:]
    y_train_np_raw, y_test_np_raw = y_np[:train_size], y_np[train_size:]

    # Aplicando reshape nos dados de treino
    X_train_reshaped = X_train_np_raw.reshape(-1, 1)
    X_test_reshaped = X_test_np_raw.reshape(-1, 1)

    # Normalização (dados de treino)
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Fit nos dados de Treino
    scaler.fit(X_train_reshaped)

    # Normalizando X train, test
    X_train_norm = scaler.transform(X_train_reshaped).reshape(X_train_np_raw.shape)
    X_test_norm = scaler.transform(X_test_reshaped).reshape(X_test_np_raw.shape)

    # Normalizando y train, test
    y_train_norm = scaler.transform(y_train_np_raw.reshape(-1, 1))
    y_test_norm = scaler.transform(y_test_np_raw.reshape(-1, 1))

    # LSTM exige a forma [batch_size, seq_length, input_size]
    # Nosso input_size é 1 (apenas o preço 'Close')
    X_train = torch.from_numpy(X_train_norm).float().to(DEVICE).unsqueeze(-1)
    y_train = torch.from_numpy(y_train_norm).float().to(DEVICE).unsqueeze(1)

    X_test = torch.from_numpy(X_test_norm).float().to(DEVICE).unsqueeze(-1)
    y_test = torch.from_numpy(y_test_norm).float().to(DEVICE).unsqueeze(1)

    # return X_train, y_train, X_test, y_test, train_size, scaler
    return X_train, y_train, X_test, y_test, scaler

    # Avaliação do Modelo


def avaliando_modelo(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        # Previsões no conjunto de teste. O X_test já está pronto para o modelo.
        test_predictions_norm = (
            model(X_test.squeeze(3)).cpu().numpy()
        )  # Traz de volta para CPU para numpy

        # Inverte a normalização para obter os preços reais
        # É CRÍTICO que o y_test esteja em [Amostras, 1] para o scaler
        y_test_pronto = y_test.view(-1, 1)

        test_predictions = scaler.inverse_transform(test_predictions_norm)
        actual_prices = scaler.inverse_transform(y_test_pronto.cpu().numpy())

    return test_predictions, actual_prices


# DataLoader para o treinamento
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------------------------------------
# Definindo parametros para o treino

params = {
    "TICKER": "ITUB4.SA",  # Ticker do Itaú na B3
    "START_DATE": "2015-01-01",  # Data inicial do dataset
    "END_DATE": "2025-05-31",  # Data final do dataset
    "DAYS_FUTURES": [1],  # Dias futuros para predição
    "TEST_SIZE": 0.2,  # 20% dos dados para teste
    "SEQ_LENGTH": [30, 45],  # Tamanho da janela de observação (dias)
    "BATCH_SIZE": [32, 64],  # Divisao em lotes de 30 ou 45 dias (SEQ_LENGTH)
    "HIDDEN_SIZE": [64, 128],  # Numero neorônios camada oculta
    "NUM_LAYERS": [2, 3],  # Numero de camadas ocultas
    "LR": [0.001, 0.002],  # Taxa de Aprendizado (Learning Rate)
    "DROPOUT_PROB": [
        0.1,
        0.2,
    ],  # Percentual de neorônios 'desativados' na camada oculta.
    "NUM_EPOCHS": [30, 50],  # vezes que o modelo irá processar o dataset de treinamento
    "LOSS_FUNCTION": ["MSE", "MAE"],  # Funções de perda (penalidade)
    "OPTIMIZER": ["Adam"],  # Otimizador
}

# -----------------------------------------------------------
# Inicio do processo de treino

# Itero sobre as estratégias
for strategy in np.arange(2, 3):

    # Download série histórica
    data = obtem_dados_historicos(
        params["TICKER"], params["START_DATE"], params["END_DATE"]
    )

    # Defino a estrategia de treino
    match strategy:
        case 1:
            data = estrategia_negociacao_1(data)
        case 2:
            data = estrategia_negociacao_2(data)
        case 3:
            data = estrategia_negociacao_3(data)
        case 4:
            data = estrategia_negociacao_4(data)
        case 5:
            data = estrategia_negociacao_5(data)
        case 6:
            data = estrategia_negociacao_6(data)

    # Número de features
    nr_features = data.shape[1]
    params["INPUT_SIZE"] = [nr_features]

    # Exporto features para CSV
    data.to_csv(f"./data/features/strategy_{strategy}.csv")
    data.values.reshape(-1, 1)

    print(
        f"\nIniciando o treinamento para estratégia {strategy} utilizando {nr_features} features...\n"
    )

    nr_modelo = 0  # Numero do modelo treinado
    dic_parametros = {}  # Dicionario para salvar parametros dos modelos

    for seq_length in params["SEQ_LENGTH"]:
        for dias_futuros in params["DAYS_FUTURES"]:

            # Construo a janela deslizante
            X, y = janelas_deslizantes_n_dias(data.to_numpy(), seq_length, dias_futuros)

            # Obtenho tamanho dados treino e teste
            test_size = params["TEST_SIZE"]
            train_size = int(len(X) * (1 - test_size))

            # Obtenho dados treino teste normalizados
            X_train, y_train, X_test, y_test, scaler = retorna_treino_teste_normalizado(
                X, y, train_size
            )

            for batch_size in params["BATCH_SIZE"]:
                for input_size in params["INPUT_SIZE"]:
                    for hidden_size in params["HIDDEN_SIZE"]:
                        for num_layers in params["NUM_LAYERS"]:
                            for lr in params["LR"]:
                                for dropout_prob in params["DROPOUT_PROB"]:
                                    for num_epochs in params["NUM_EPOCHS"]:
                                        for loss_function in params["LOSS_FUNCTION"]:

                                            # Carrego o DataLoader para treinamento
                                            train_dataset = StockDataset(
                                                X_train, y_train
                                            )
                                            train_loader = DataLoader(
                                                train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                            )

                                            # Inicializa o modelo
                                            model = SimpleLSTM(
                                                input_size=input_size,
                                                hidden_size=hidden_size,
                                                num_layers=num_layers,
                                                output_size=1,
                                                dropout_prob=dropout_prob,
                                            ).to(DEVICE)

                                            # Defino Função de Perda
                                            if loss_function == "MSE":
                                                criterion = nn.MSELoss()
                                            elif loss_function == "MAE":
                                                criterion = nn.L1Loss()

                                            # Defino o otimizador
                                            optimizer = torch.optim.Adam(
                                                model.parameters(), lr=lr
                                            )

                                            # Treinando o modelo

                                            # Incremento numero do modelo
                                            nr_modelo = nr_modelo + 1

                                            print(f"Treinando modelo {nr_modelo}..")

                                            # Obtenho data/hora inicio treinamento do modelo
                                            data_inicial = time.localtime()
                                            data_inicial = time.strftime(
                                                "%Y-%m-%d %H:%M:%S", data_inicial
                                            )

                                            for epoch in range(num_epochs):
                                                model.train()

                                                for batch_X, batch_y in train_loader:

                                                    # batch_X e batch_y JÁ estão no dispositivo (DEVICE)
                                                    batch_X = batch_X.squeeze(3)
                                                    batch_y = batch_y.view(-1, 1)

                                                    # Forward pass
                                                    outputs = model(batch_X)
                                                    loss = criterion(outputs, batch_y)

                                                    # Backward and optimize
                                                    optimizer.zero_grad()
                                                    loss.backward()
                                                    optimizer.step()

                                                if (epoch + 1) % 10 == 0:
                                                    print(
                                                        f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}"
                                                    )

                                            # Obtenho data/hora final do treinamento do modelo
                                            data_final = time.localtime()
                                            data_final = time.strftime(
                                                "%Y-%m-%d %H:%M:%S", data_final
                                            )

                                            # Avaliando o modelo treinado

                                            # Obtendo previsões no conjunto de teste
                                            test_predictions, actual_prices = (
                                                avaliando_modelo(model, X_test, y_test)
                                            )

                                            # Medindo Erro Absoluto Médio (MAE)
                                            mae = mean_absolute_error(
                                                actual_prices, test_predictions
                                            )

                                            # Medindo o Erro Quadrático Médio (MSE)
                                            mse = mean_squared_error(
                                                actual_prices, test_predictions
                                            )

                                            # Medindo o Erro Quadratico Médio da Regressão (RMSE)
                                            rmse = root_mean_squared_error(
                                                actual_prices, test_predictions
                                            )

                                            # Medindo a Acurácia Direcional (DAC)
                                            dac = np.mean(
                                                (
                                                    np.sign(
                                                        actual_prices[1:]
                                                        - actual_prices[:-1]
                                                    )
                                                    == np.sign(
                                                        test_predictions[1:]
                                                        - test_predictions[:-1]
                                                    )
                                                ).astype(int)
                                            )

                                            # Salvando o modelo e os resultados

                                            try:
                                                # Caminho completo da pasta (no diretório atual)
                                                caminho_pasta = (
                                                    f"./models/strategy_{strategy}"
                                                )

                                                # Verifica se a pasta já existe
                                                if not os.path.exists(caminho_pasta):
                                                    os.makedirs(caminho_pasta)

                                                # Saida modelo .pkl
                                                nome_arquivo = (
                                                    caminho_pasta
                                                    + f"/modelo_lstm_{str(nr_modelo)}.pkl"
                                                )

                                                # Salvo arquivo .pkl do modelo
                                                with open(
                                                    nome_arquivo, "wb"
                                                ) as arquivo:
                                                    pickle.dump(model, arquivo)

                                                    # Gravo os dados do experimento
                                                    dic_parametros[nr_modelo] = {
                                                        "nr_model": nr_modelo,
                                                        "seq_length": seq_length,
                                                        "batch_size": batch_size,
                                                        "hidden_size": hidden_size,
                                                        "num_layers": num_layers,
                                                        "lr": lr,
                                                        "dropout_prob": dropout_prob,
                                                        "num_epochs": num_epochs,
                                                        "loss_function": loss_function,
                                                        "data_inicial": data_inicial,
                                                        "data_final": data_final,
                                                        "dias_futuros": dias_futuros,
                                                        "mae": mae,
                                                        "mse": mse,
                                                        "rmse": rmse,
                                                        "dac": dac,
                                                    }

                                                    # Salvo .csv experiments
                                                    df_experiments = pd.DataFrame(
                                                        dic_parametros
                                                    )
                                                    df_experiments = df_experiments.T
                                                    df_experiments.set_index(
                                                        "nr_model", inplace=True
                                                    )
                                                    df_experiments.to_csv(
                                                        f"./reports/report_strategy_{strategy}.csv"
                                                    )

                                                    print(
                                                        f"Treinamento concluído. Modelo salvo com sucesso em: {nome_arquivo}\n"
                                                    )

                                            except Exception as e:
                                                print(
                                                    f"Erro ao salvar o modelo {str(nr_modelo)}: {e}\n"
                                                )
