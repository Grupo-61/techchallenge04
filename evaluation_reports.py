import pandas as pd
import numpy as np

# Itero sobre os relatórios das estratégias 1, 2 e 3
for strategy in np.arange(1, 4):

    print(f"Avaliando estratégia {strategy}...")

    # Carrego o arquivo CSV correspondente à estratégia 
    data = pd.read_csv(f"./reports/all_params/all_report_strategy_{strategy}.csv")    

    # Obtenho mae minimo
    mae= data.loc[data['mae'] == data['mae'].min()]

    # Obtenho o mse minimo
    mse= data.loc[data['mse'] == data['mse'].min()]

    # Obtenho o rmse minimo
    rmse= data.loc[data['rmse'] == data['rmse'].min()]
                      
    # Obtenho dac maximo
    dac= data.loc[data['dac'] == data['dac'].max()]

    # Concateno os resultados em um único DataFrame
    df_eval = pd.concat([mae, mse, rmse, dac]).drop_duplicates().reset_index(drop=True)

    # Salvo o arquivo CSV com os melhores resultados
    df_eval.to_csv(f"./reports/best_params/best_report_strategy_{strategy}.csv", index=False) 
    
