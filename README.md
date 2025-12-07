# Tech Challenge 4 (Fase 4): Construção de Modelos de Machine Learning usando redes neurais LSTM (Long Short-Term Memory) para a previsão de preços de ativos financeiros.

Para o Tech Challenge 4, o desafio proposto foi o seguinte:


📢 **Problema:**

Criar um modelo preditivo de redes neurais Long Short Term Memory (LSTM) para predizer o valor de fechamento da bolsa de valores de uma empresa à sua escolha e realizar toda a pipeline de desenvolvimento, desde a criação do modelo preditivo até o deploy do modelo em uma API que permita a previsão de preços de ações.


## 📌 Objetivos

- Utilizar a biblioteca yfinance do Yahoo finance para baixar a série histórica de um ativo financeiro;
- Tratar e normalizar os dados para treinar e avaliar o modelo LSTM;
- Implementar um modelo de Deep Learning usando LSTM para capturar padrões temparais nos dados de preço das ações, já devidamente tratados;
- Treinar o modelo utilizando parte dos dados tratados e ajuste dos hiperparâmetros para otimizar o desempenho;
- Avaliar o modelo utilizando dados de validação, utilizando métricas como MAE (Mean Absolute Error), RMSE (Root Mean Square Error), MAPE (Erro Percentual Absoluto Médio) ou outra métrica apropriada para medir a precisão das previsões;
- Salvar o modelo com desempenho satisfatório para ser utilizado futuramente para inferência;
- Deploy do Modelo desenvolvendo uma API RESTFull utilizando Flask ou FastAPI para servir o modelo. A API deve permitir que o usuário forneça dados históricos de preços e receba previsões dos preços
futuros;
- Configurar ferramentas de monitoramento para rastrear a performance do modelo em produção, incluindo tempo de resposta e utilização de recursos.


## 📌 Entregáveis
- Código-fonte do modelo LSTM no seu repositório do GIT + documentação do projeto.
- Scripts ou contêineres Docker para deploy da API.
- Link para a API em produção, caso tenha sido deployada em um ambiente de nuvem.

  
## Possíveis dores


- Definição de um bom dataset que atenda aos requisitos de qualidade para a produção de bons modelos de Machine Learning;
- Baixa qualidade dos dados com problemas de valores faltantes (missing values) e registros inconsistentes;
- Grande volume de dados, o que pode tornar o processamento lento e caro;
  

## Proposta de solução


A fim de garantir a qualidade dos dados, optamos por utilizar o Índice Bovespa Smart Low Volatility B3 (Ibov Smart Low Vol B3) como critério para a seleção dos ativos, uma vez que, seu objetivo é ser o
indicador de desempenho médio dos ativos de maior negociabilidade, representatividade e que possuem menor volatilidade nos retornos diários. 

Através de uma análise de EDA realizada nesta carteira teórica de indice, optamos por selecionar o banco ITAÚ (ITUB4) devido a sua grande base histórica (mais de 10 anos de negociação ininterrupta), bem como, baixa voltilidade.

[Índice Bovespa Smart Low Volatility B3 (Metodologia)](https://www.b3.com.br/data/files/A5/34/90/0A/E28B09105FE89209AC094EA8/Metodologia_IbovLowVolB3_PT.pdf) 


**Importante**


O arquivo nn_eda.ipynb contém uma análise EDA enxuta construída para seleção deste ativo, porém é importante frizar que outros ativos também estavam aptos para escolha. Toda a implementação foi feita usando **Python e bibliotecas**, tais como:

- Dados, tratamento e análise: Yfinance, Pandas e Numpy
- Gráficas: MatplotLib
- Machine Learning: Sklearn, Pytorch
- Outras: os, time, pickle 


### 📂 Estrutura do projeto


```
.
├── data
│   └── features                # Armazena os dados brutos e arquivos .csv processados
├── models                      # Armazena os modelos .plk treinados
├── reports                     # Relatórios de performance dos modelos
│   ├── report_strategy_X.csv   # Logs com parâmetros e métricas de todos os treinos
│   └── best_strategy_X.csv     # Relatório filtrado com os melhores modelos
└── src                         # Código fonte e scripts de execução
    ├── strategy.py             # Definição das estratégias de Feature Engineering
    ├── nn_lstm.py              # Script principal para treinamento da rede LSTM
    ├── evaluation_reports.py   # Cria o relatório bes_strategy_X.csv organizando os melhores modelos e parametros
    └── evaluation_models.py    # Le bes_strategy_X.csv carregando o melhor modelo e avalia nos dados de teste
```


## ⚙️ Instalação e Configuração
Para reproduzir este projeto, é necessário preparar o ambiente Python. Recomenda-se o uso de um ambiente virtual (venv ou conda).


1. Clonar e preparar o ambiente


## Crie e ative um ambiente virtual (exemplo com venv)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate


## Instale as dependências

```
pip install -r requirements.txt
```

**Nota sobre GPU:** Certifique-se de que você possui os drivers da NVIDIA instalados e a versão do PyTorch compatível com o seu CUDA toolkit para habilitar a aceleração via GPU. Caso sua placa de vídeo Nvidia seja mais antiga talvez seja necessário instalar uma versão do Pytorch mais antiga. Como opção, teste o seguinte comando:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 🚀 Como Executar (Reprodução)
O pipeline deve ser executado seguindo a ordem lógica dos scripts contidos na pasta src.


## 1. Treinamento do Modelo (GPU)
Execute o script principal para treinar a rede LSTM. O script detectará automaticamente se uma GPU (CUDA) está disponível para acelerar o treinamento.

```
python src/nn_lstm.py
```
- Saída: Os modelos treinados serão salvos automaticamente na pasta /models.
- Logs: Os parâmetros e métricas de cada época/experimento são salvos em /reports.


## 2. Seleção do Melhor Modelo
Após o treinamento, analise os relatórios para filtrar as melhores performances.


```
python src/evaluation_reports.py
```

**Nota:** Isso gerará um arquivo (ex: best_strategy_2.csv) ordenando os modelos do melhor para o pior resultado.


## 3. Avaliação Final
Carregue o melhor modelo salvo em /models e realize a inferência nos dados de teste (dados nunca vistos pela rede).

```
python src/evaluation_models.py
```

## 📊 Resultados
O modelo é avaliado utilizando as métricas MAE, MSE e DAC. Atualmente, o melhor modelo configurado alcançou:


Acurácia Direcional (DAC): > 75%

## 🛠️ Tecnologias
- Linguagem: Python 3.x
- Deep Learning: PyTorch (Suporte a CUDA/GPU)
- Manipulação de Dados: Pandas, NumPy
- Métricas & Preprocessamento: Scikit-Learn


## Vídeo de Apresentação no Youtube (Modelo LSTM)
Para melhor compreensão da entrega , foi produzido um vídeo de apresentação no Youtube:

[Link para a Vídeo](https://youtu.be/_WMI-M6gzXY)


## ✒️ Autores

| Nome                            |   RM    | Link do GitHub                                      |
|---------------------------------|---------|-----------------------------------------------------|
| Ana Paula de Almeida            | 363602  | [GitHub](https://github.com/Ana9873P)               |
| Augusto do Nascimento Omena     | 363185  | [GitHub](https://github.com/AugustoOmena)           |
| Bruno Gabriel de Oliveira       | 361248  | [GitHub](https://github.com/brunogabrieldeoliveira) |
| José Walmir Gonçalves Duque     | 363196  | [GitHub](https://github.com/WALMIRDUQUE)            |

## 📄 Licença

Este projeto está licenciado sob a Licença MIT.  
Consulte o arquivo [license](docs/license/license.txt)  para mais detalhes.
