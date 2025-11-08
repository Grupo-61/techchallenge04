# Instalação do PyTorch com suporte CUDA (GPU) NVIDIA MX150, CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Importações necessárias
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import matplotlib.pyplot as plt
import pickle
import time

# Verificar a versão do PyTorch e a disponibilidade da GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")

print('\n')

###########################
# FUNÇÕES AUXILIARES
###########################

# Obtem dados historicos do Yahoo Finance
def obtemDadosHistoricos(ticker, data_inicial, data_final):
    dados= yf.download(ticker, start=data_inicial, end=data_final)
    colunas= []
    for col in dados.columns:
        colunas.append(col[0])
    dados.columns= colunas
    return dados

# Calcula retorno diario
def retornoDiario(precos):
    retornos= precos.pct_change()
    retornos= retornos.dropna()
    return retornos

# Calcula voltatilidade ultimos n dias
def volatilidadeUltimosNDias(retornos_diarios, n):
    retornos_recentes= retornos_diarios.tail(n)
    return np.std(retornos_recentes) * np.sqrt(252)

# Construindo a janela deslizante
def create_sequences_multivariate(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)


# Construindo features

# Estratégia 1: Apenas preço de fechamento
def build_features_1(data):
    columns= ['Close']
    return data[columns]

# Estratégia 2: Preço de abertura, máxima, mínima, fechamento e volume
def build_features_2(data):
    columns= ['Open', 'High', 'Low', 'Close']
    return data[columns]

# Estratégia 3: Preço de fechamento, volume, retorno diário e volatilidade dos últimos 5 dias
def build_features_3(data):
    data['daily_return'] = data['Close'].pct_change()
    data['5-day_volatility'] = data['daily_return'].rolling(window=5).apply(lambda x: volatilidadeUltimosNDias(x, 5))
    columns= ['Close', 'daily_return', '5-day_volatility']
    return data[columns]

# Retorna dados treino e teste tratados/normalizados
def return_train_test(data, seq_length, test_size):

  # Construindo a janela deslizante
  data = data.to_numpy()
  X_np, y_np = create_sequences_multivariate(data, seq_length)

  # Divide as sequências em Treino e Teste
  train_size = int(len(X_np) * (1 - test_size))
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

  return X_train, y_train, X_test, y_test, train_size, scaler

# Salvando Grafico predição
def save_graph(ticker, data, caminho, seq_length, train_size, test_predictions, actual_prices):
  
  # Plotagem
  plt.figure(figsize=(16, 9))
  plt.plot(data.index[train_size + seq_length:], actual_prices, label='Preço Real', color='blue')
  plt.plot(data.index[train_size + seq_length:], test_predictions, label='Previsão LSTM', color='red')
  plt.title(f'Previsão de Preço de Fechamento do {ticker} (LSTM)')
  plt.xlabel('Data')
  plt.ylabel('Preço (R$)')
  plt.legend()
  plt.grid(True)
  plt.savefig(caminho, bbox_inches='tight')
  plt.close()

  # Avaliação do Modelo
def model_evaluation(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        # Previsões no conjunto de teste. O X_test já está pronto para o modelo.
        test_predictions_norm = model(X_test.squeeze(3)).cpu().numpy() # Traz de volta para CPU para numpy

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
    
# Definição do Modelo LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # input_size=5 ('Close', 'daily_return', '5-day_volatility', '10-day_volatility', '15-day_volatility')
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Camanda dropout (desliga neuronios)
        self.dropout = nn.Dropout(dropout_prob)

        # Conecta a saída do LSTM ao output final (previsão de 1 valor)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Inicializa hidden state (h0) e cell state (c0) no DEVICE
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)

        # Passa o input pelo LSTM
        # out: [batch_size, seq_length, hidden_size]
        out, _ = self.lstm(x, (h0, c0))

        # Pegamos apenas a saída do último passo da sequência (last time step)
        # out[:, -1, :] tem a forma [batch_size, hidden_size]
        out = self.fc(out[:, -1, :])
        return out

# PARAMETOS PARA COLETA DE DADOS E TREINAMENTO
params= {
    'TICKER': 'ITUB4.SA',             # Ticker do Itaú na B3    
    'START_DATE': '2015-01-01',       # Data inicial do dataset
    'END_DATE': '2025-05-31',         # Data final do dataset
    'TEST_SIZE': 0.2,                 # 20% dos dados para teste
    'SEQ_LENGTH': [30,45],            # Tamanho da janela de observação (dias)
    'BATCH_SIZE': [32,64],            # Divisao em lotes de 30 ou 45 dias (SEQ_LENGTH)
    'HIDDEN_SIZE': [32,64],           # Numero neorônios camada oculta
    'NUM_LAYERS': [2, 3],             # Numero de camadas ocultas
    'LR': [0.001,0.002],              # Taxa de Aprendizado (Learning Rate)
    'DROPOUT_PROB': [0.1,0.2],        # Percentual de neorônios 'desativados' na camada oculta.
    'NUM_EPOCHS': [30, 50],           # vezes que o modelo irá processar o dataset de treinamento
    'LOSS_FUNCTION': ['MSE', 'MAE'],  # Funções de perda (penalidade)
    'OPTIMIZER': ['Adam', 'RMSprop'], # Otimizador
    }

# Itero sobre as estratégias
for strategy in np.arange(1, 4):

    # COLETA E PREPARAÇÃO DOS DADOS
        
    # Obtenho dados históricos
    data = obtemDadosHistoricos(params['TICKER'], params['START_DATE'], params['END_DATE'])

    # CONSTRUÇÃO DAS FEATURES SEGUNDO A ESTRATÉGIA ESCOLHIDA

    # Estratégia 1: Apenas preço de fechamento
    if strategy == 1:
       data = build_features_1(data)
    
    # Estratégia 2: Preço de abertura, máxima, mínima, fechamento e volume
    elif strategy == 2:
       data = build_features_2(data)
       
    # Estratégia 3: Preço de fechamento, volume, retorno diário e volatilidade dos últimos 5 dias
    elif strategy == 3:
       data = build_features_3(data)
       data= data[5:]

    # Defino número de features
    nr_features= data.shape[1]
    params['INPUT_SIZE']= [nr_features]

    # Exportando features para CSV    
    data.to_csv(f'./features/features_strategy_{strategy}.csv')
    data.values.reshape(-1, 1)

    print(f'\nIniciando treinamento para estratégia {strategy} com {nr_features} features...\n')
    print('\n')

    # TREINAMENTO E AVALIAÇÃO DO MODELO

    nr_modelo= 0        # Numero do modelo treinado
    dic_parametros= {}  # Dicionario para salvar parametros dos modelos
    caminho= './'       # Caminho para salvar modelos e experimentos

    # Itero sobre seq_length
    for seq_length in params['SEQ_LENGTH']:
      
      # Defino test_size
      test_size= params['TEST_SIZE']

      # Carrego dados de treino e teste tratados
      X_train, y_train, X_test, y_test, train_size, scaler = return_train_test(data, seq_length, test_size)

      # Itero sobre batch_size
      for batch_size in params['BATCH_SIZE']:

        # Itero sobre input_size
        for input_size in params['INPUT_SIZE']:

          # Itero sobre hidden_size
          for hidden_size in params['HIDDEN_SIZE']:

            # Itero sobre num_layers
            for num_layers in params['NUM_LAYERS']:

              # Itero sobre LR
              for lr in params['LR']:

                # Itero sobre dropout_prob
                for dropout_prob in params['DROPOUT_PROB']:

                  # Itero sobre num_epochs
                  for num_epochs in params['NUM_EPOCHS']:

                    # Itero sobre loss_function
                    for loss_function in params['LOSS_FUNCTION']:

                      # Carrego o DataLoader para treinamento
                      train_dataset = StockDataset(X_train, y_train)
                      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                      # Inicializa o modelo
                      model = SimpleLSTM(input_size= input_size, 
                                         hidden_size= hidden_size, 
                                         num_layers= num_layers,
                                         output_size= 1,
                                         dropout_prob= dropout_prob)\
                                          .to(DEVICE)

                      # Defino Função de Perda
                      if loss_function == 'MSE':
                        criterion = nn.MSELoss()
                      elif loss_function == 'MAE':
                        criterion = nn.L1Loss()

                      # Defino o otimizador
                      optimizer = torch.optim.Adam(model.parameters(), lr=lr)                      

                      # TREINAMENTO DO MODELO

                      print(f'Iniciando treino com seq_length: {seq_length}, batch_size: {batch_size}, hidden_size: {hidden_size}, num_layers: {num_layers}, lr: {lr}, dropout_prob: {dropout_prob}, num_epochs: {num_epochs}, loss_function: {loss_function}')

                      # Obtenho data/hora inicio treinamento do modelo
                      data_inicial = time.localtime()
                      data_inicial = time.strftime("%Y-%m-%d %H:%M:%S", data_inicial)

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
                              print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

                      # Obtenho data/hora final do treinamento do modelo
                      data_final = time.localtime()
                      data_final = time.strftime("%Y-%m-%d %H:%M:%S", data_final)

                      # AVALIAÇÃO DO MODELO

                      # Obtendo previsões no conjunto de teste
                      test_predictions, actual_prices = model_evaluation(model, X_test, y_test)

                      # Medindo Erro Absoluto Médio (MAE)
                      mae = mean_absolute_error(actual_prices, test_predictions)

                      # Medindo o Erro Quadrático Médio (MSE)
                      mse = mean_squared_error(actual_prices, test_predictions)

                      # Medindo o Erro Quadratico Médio da Regressão (RMSE)
                      rmse = root_mean_squared_error(actual_prices, test_predictions)

                      # Medindo a Acurácia Direcional (DAC)
                      dac = np.mean((np.sign(actual_prices[1:] - actual_prices[:-1]) == np.sign(test_predictions[1:] - test_predictions[:-1])).astype(int))

                      # SALVANDO O MODELO E RESULTADOS

                      try:
                        # Defino o numero do modelo
                        nr_modelo= nr_modelo + 1

                        # Definindo nome modelo .pkl
                        nome_arquivo = caminho + f'experiments/strategy_{strategy}/models/modelo_lstm_{str(nr_modelo)}.pkl'

                        with open(nome_arquivo, 'wb') as arquivo:

                          # Salvo arquivo .pkl do modelo
                          pickle.dump(model, arquivo)

                          # Gravo os dados do experimento
                          dic_parametros[nr_modelo] = {
                              'nr_model': nr_modelo,
                              'seq_length': seq_length,
                              'batch_size': batch_size,
                              'hidden_size': hidden_size,
                              'num_layers': num_layers,
                              'lr': lr,
                              'dropout_prob': dropout_prob,
                              'num_epochs': num_epochs,
                              'loss_function': loss_function,
                              'data_inicial': data_inicial,
                              'data_final': data_final,
                              'mae': mae,
                              'mse': mse,
                              'rmse': rmse,
                              'dac': dac
                          }

                          # Salvo .csv experiments
                          df_experiments = pd.DataFrame(dic_parametros)
                          df_experiments = df_experiments.T
                          df_experiments.set_index('nr_model', inplace=True)
                          df_experiments.to_csv(caminho + f'reports/report_strategy_{strategy}_lstm.csv')

                          print(f"Treinamento concluído. Modelo salvo com sucesso em: {nome_arquivo}")
                          print("\n")

                      except Exception as e:
                        print(f'Erro ao salvar o modelo {str(nr_modelo)}: {e}')
                        print("\n")