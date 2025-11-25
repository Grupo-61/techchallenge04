import numpy as np
import pandas as pd

# Funções de apoio

# Calcula retorno diario
def retornoDiario(precos):
    retornos = precos.pct_change()
    retornos = retornos.dropna()
    return retornos

# Calcula voltatilidade ultimos n dias
def volatilidadeUltimosNDias(retornos_diarios, n):
    retornos_recentes= retornos_diarios.tail(n)
    return np.std(retornos_recentes) * np.sqrt(252)

# Estratégia 1: Apenas preço de fechamento
def estrategiaNegociacao1(data):
    df = data.copy()
    columns = ['Close']
    return df[columns]

# Estratégia 2: Preço de abertura, máxima, mínima, fechamento e volume
def estrategiaNegociacao2(data):
    df = data.copy()
    columns = ['Open', 'High', 'Low', 'Close']
    return df[columns]

# Estratégia 3: Preço de fechamento, retorno diário e volatilidade
def estrategiaNegociacao3(data):
    df = data.copy()
    df['daily_return'] = df['Close'].pct_change()
    df['5-day_volatility'] = df['daily_return'].rolling(window=5).std() * np.sqrt(252)    
    columns = ['Close', 'daily_return', '5-day_volatility']
    return df[columns].dropna()

# Estratégia 4: Tendência (Médias Móveis)
def estrategiaNegociacao4(data):
    df = data.copy()
    
    # Média Móvel Simples (SMA) de 7 dias (Curto prazo)
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    
    # Média Móvel Exponencial (EMA) de 21 dias (Médio prazo)
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    columns = ['Close', 'SMA_7', 'EMA_21']
    return df[columns].dropna()

# Estratégia 5: Momentum (RSI e MACD)
def estrategiaNegociacao5(data):
    df = data.copy()
    
    # --- Cálculo do RSI (Relative Strength Index) ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # --- Cálculo do MACD (Moving Average Convergence Divergence) ---
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    columns = ['Close', 'RSI_14', 'MACD', 'MACD_Signal']
    return df[columns].dropna()

# Estratégia 6: Volatilidade e Faixa de Preço (Bandas de Bollinger)
def estrategiaNegociacao6(data):
    df = data.copy()
    
    # Janela de 20 dias é padrão para Bollinger
    periodo = 20
    sma = df['Close'].rolling(window=periodo).mean()
    std = df['Close'].rolling(window=periodo).std()
    
    # Banda Superior e Inferior
    df['Bollinger_Upper'] = sma + (std * 2)
    df['Bollinger_Lower'] = sma - (std * 2)
    
    # Largura da banda (indica volatilidade compressão/expansão)
    df['Bollinger_Width'] = df['Bollinger_Upper'] - df['Bollinger_Lower']
    
    columns = ['Close', 'Bollinger_Upper', 'Bollinger_Lower', 'Bollinger_Width']
    return df[columns].dropna()