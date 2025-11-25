import pandas as pd
import numpy as np

# Itero sobre os relatórios das estratégias 1, 2 e 3
for strategy in np.arange(2, 3):

    print(f"Avaliando estratégia {strategy}...")

    # Carrego o arquivo CSV correspondente à estratégia 
    data = pd.read_csv(f"./reports/report_strategy_{strategy}.csv")    

    # Exporto DAC maior igual a 50%
    data= data.loc[data['dac'] >= 0.5]

    # Organiza MAE do menor para o maior
    data= data.sort_values(by='mae')

    # Exporto CSV
    data.to_csv(f"./reports/best_strategy_{strategy}.csv", index=False) 


    
