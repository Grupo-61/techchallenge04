import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
import io

# Verificar a versão do PyTorch e a disponibilidade da GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")

# FUNÇÕES AUXILIARES    

# Obtem dados historicos do Yahoo Finance
def obtemDadosHistoricos(ticker, data_inicial, data_final):
    dados= yf.download(ticker, start=data_inicial, end=data_final)
    colunas= []
    for col in dados.columns:
        colunas.append(col[0])
    dados.columns= colunas
    return dados

# Estratégia 2: Preço de abertura, máxima, mínima, fechamento e volume
def build_features_2(data):
    columns= ['Open', 'High', 'Low', 'Close']
    return data[columns]

# Construindo a janela deslizante
def create_sequences_multivariate(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

def janelaDeslizanteNDias(data, seq_length, dias_futuros=1):
    X, y = [], []    
    limite = len(data) - seq_length - dias_futuros + 1    
    for i in range(limite):
        X.append(data[i : i + seq_length, :])        
        target_index = i + seq_length + dias_futuros - 1
        y.append(data[target_index, 0]) # Assume que o alvo é a coluna 0 (Close, geralmente)
        
    return np.array(X), np.array(y)

# Definição do Modelo LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
    
# PARÂMETROS GERAIS

TICKER = 'ITUB4.SA'       # Ticker do Itaú na B3
START_DATE = '2025-06-01' # Data inicial do dataset
END_DATE = '2025-10-31'   # Data final do dataset
NR_STRATEGY = 2           # Número da estratégia de features utilizada

# Obtendo modelo ótimo salvo
model_path= f'./reports/best_strategy_{NR_STRATEGY}.csv'
model_df= pd.read_csv(model_path)
NR_MODEL= int(model_df.iloc[0]['nr_model'])

# Lendo numero de features do modelo salvo
features= f'./data/features/strategy_{NR_STRATEGY}.csv'
features_df= pd.read_csv(features)
input_size= len(features_df.columns) -1 # Subtrai 1 para não contar a coluna Date

# Lendo parâmetros do modelo salvo
report= f'./reports/report_strategy_{NR_STRATEGY}.csv'
params_df= pd.read_csv(report)
params_df= params_df.iloc[NR_MODEL -1] 

# Extraindo parâmetros
seq_length = int(params_df['seq_length']) 
hidden_size= int(params_df['hidden_size'])
num_layers= int(params_df['num_layers'])
dropout_prob= float(params_df['dropout_prob'])
dias_futuros= int(params_df['dias_futuros'])

print(f'\nModelo: {NR_MODEL} | seq_length: {seq_length} | hidden_size: {hidden_size} | num_layers: {num_layers} | dropout_prob: {dropout_prob}')

# COLETA E PREPARAÇÃO DOS DADOS

print("\nObtendo dados de validação...")

# Obtendo dados de validação
data = obtemDadosHistoricos(TICKER, START_DATE, END_DATE)
data = build_features_2(data)

# Construindo a janela deslizante
data = data.to_numpy()

X, y= janelaDeslizanteNDias(data, seq_length, dias_futuros)
X_test_reshaped = X.reshape(-1, 1)
y_test_reshaped = y.reshape(-1, 1)

# Normalização nos dados de validação
scaler = MinMaxScaler(feature_range=(-1, 1))

# Fit nos dados de validação
scaler.fit(X_test_reshaped)

# Normalizando dados de validacao
X_test_norm = scaler.transform(X_test_reshaped).reshape(X.shape)
y_test_norm = scaler.transform(y.reshape(-1, 1))

# Convertendo para tensores do PyTorch
X_test = torch.from_numpy(X_test_norm).float().to(DEVICE).unsqueeze(-1)
y_test = torch.from_numpy(y_test_norm).float().to(DEVICE).unsqueeze(1)

# CARREGANDO MODELO TREINADO

print("\nCarregando modelo treinado...")

# Definindo o caminho do modelo salvo
arquivo_modelo = f'./models/strategy_2/modelo_lstm_{NR_MODEL}.pkl'

# Caso GPU disponível
if torch.cuda.is_available():
    try:
        with open(arquivo_modelo, 'rb') as f:
            model = pickle.load(f)

        print("Modelo carregado com sucesso na GPU.")
        # Se for um modelo PyTorch, lembre-se de colocá-lo no modo de avaliação
        if hasattr(model, 'eval'):
            model.eval()

    except Exception as e:
        print(f"Erro ao carregar modelo na GPU: {e}")

# Caso GPU não disponível - Forçar carregamento na CPU
else:

    # CPU - Custom Unpickler para forçar o carregamento na CPU
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                # Retorna uma função lambda que chama torch.load forçando a CPU
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                # Para o resto, usa o comportamento padrão
                return super().find_class(module, name)

    # --- Como usar ---
    try:
        with open(arquivo_modelo, 'rb') as f:
            # Usa o Unpickler personalizado para carregar o objeto
            model = CPU_Unpickler(f).load()

        print("Modelo carregado com sucesso na CPU usando CPU_Unpickler.")
        # Se for um modelo PyTorch, lembre-se de colocá-lo no modo de avaliação
        if hasattr(model, 'eval'):
            model.eval()

    except Exception as e:
        print(f"Erro ao carregar com CPU_Unpickler: {e}")


# AVALIAÇÃO DO MODELO

print("Plotando gráfico para acompanhar desempenho...")

# Plotando previsões vs valores reais
model.eval()
with torch.no_grad():
    # Previsões no conjunto de teste. O X_test já está pronto para o modelo.
    test_predictions_norm = model(X_test.squeeze(3)).cpu().numpy() # Traz de volta para CPU para numpy

    # Inverte a normalização para obter os preços reais
    # É CRÍTICO que o y_test esteja em [Amostras, 1] para o scaler
    y_test_pronto = y_test.view(-1, 1)

    test_predictions = scaler.inverse_transform(test_predictions_norm)
    actual_prices = scaler.inverse_transform(y_test_pronto.cpu().numpy())

saida_plot = f'./reports/plots/graph_strategy_2_lstm.png'

# Plotagem
plt.figure(figsize=(16, 9))
plt.plot(pd.DataFrame(actual_prices).index[:], pd.DataFrame(actual_prices), label='Preço Real', color='blue')
plt.plot(pd.DataFrame(test_predictions).index[:], pd.DataFrame(test_predictions), label='Previsão LSTM', color='red')
plt.title(f'Previsão de Preço de Fechamento do {TICKER} (LSTM)')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig(saida_plot, bbox_inches='tight')
plt.close()

# AVALIANDO MÉTRICAS DE ERRO
print("\nAvaliando métricas de erro..")

# Medindo Erro Absoluto Médio (MAE)
mae = mean_absolute_error(actual_prices, test_predictions)
print(f"\nMAE no Conjunto de Validação: R$ {mae:.4f}")

# Medindo o Erro Quadrático Médio (MSE)
mse = mean_squared_error(actual_prices, test_predictions)
print(f"MSE no Conjunto de Validação: R$ {mse:.4f}")

# Medindo o Erro Quadratico Médio da Regressão (RMSE)
rmse = root_mean_squared_error(actual_prices, test_predictions)
print(f"RMSE no Conjunto de Validação: R$ {rmse:.4f}")

# Medindo a Acurácia Direcional (DAC)
dac = np.mean((np.sign(actual_prices[1:] - actual_prices[:-1]) == np.sign(test_predictions[1:] - test_predictions[:-1])).astype(int))
print(f"DAC no Conjunto de Validação: {dac:.2%}")













