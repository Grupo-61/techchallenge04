import pandas as pd
import numpy as np

# Itero sobre os relatórios das estratégias 1, 2 e 3
for strategy in np.arange(1, 4):

    print(f"Avaliando estratégia {strategy}...")

    try:
        # Carrego o arquivo CSV correspondente à estratégia
        data = pd.read_csv(f"./reports/report_strategy_{strategy}.csv")

        # Exporto DAC maior igual a 8%
        data = data.loc[data["dac"] >= 0.8]

        # Organiza MAE do menor para o maior
        data = data.sort_values(by="mae")

        # Exporto CSV
        data.to_csv(f"./reports/best_strategy_{strategy}.csv", index=False)

    except FileNotFoundError:
        print(
            f"Arquivo para a estratégia {strategy} não encontrado. Avaliando próxima estratégia..."
        )
        continue
