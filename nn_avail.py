import pandas as pd

# Carregar os dados dos experimentos
df_experimentos = pd.read_csv('reports/experimentos_lstm.csv')

# Exibir dados mae minimo
mae= df_experimentos[df_experimentos['mae'] == df_experimentos['mae'].min()]
print(mae)

# Exibir dados mse minimo
mse= df_experimentos[df_experimentos['mse'] == df_experimentos['mse'].min()]
print(mse)

# Exibir dados rmse minimo
rmse= df_experimentos[df_experimentos['rmse'] == df_experimentos['rmse'].min()]
print(rmse)

# Exibir dados dac maximo
dac= df_experimentos[df_experimentos['dac'] == df_experimentos['dac'].max()]
print(dac)