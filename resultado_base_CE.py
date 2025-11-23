import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

url = 'Base de Dados/Base Pre Processada/BASE_CE_PRE_PROCESSADA.csv'
dados_escola = pd.read_csv(url, sep=',')

# Remove coluna
dados_escola = dados_escola.iloc[:, 1:]
print(dados_escola)


###1.3. Quantidade de valores Nulos
quantidade_nulos = dados_escola.isnull().sum()
print(quantidade_nulos[quantidade_nulos > 0])

#1.5. Remove Colunas
##1.5.1 Remove Coluna "Unnamed: 0"
if "Unnamed: 0" in dados_escola.columns:
    dados_escola.drop(columns=["Unnamed: 0"], inplace=True)

####1.5.2 Remove Colunas irrelevantes
#####1.5.2.1 Remove Coluna "ID_ESCOLA", código de cada escola
dados_escola.drop(columns=["ID_ESCOLA"], inplace=True)

#####1.5.2.2 Remove Colunas que tem apenas um valor
#Lista colunas com um único valor
colunas_unico_valor = [col for col in dados_escola.columns if dados_escola[col].nunique() == 1]
print(colunas_unico_valor)

# Remove as colunas com um único valor
dados_escola = dados_escola.drop(columns=colunas_unico_valor)
print(dados_escola)

###1.6. Normaliza os valores
# Crie um objeto MinMaxScaler
scaler = MinMaxScaler()

# Selecione as colunas numéricas para normalização
numeric_cols = dados_escola.select_dtypes(include=np.number).columns

# Aplique a normalização às colunas numéricas
dados_escola[numeric_cols] = scaler.fit_transform(dados_escola[numeric_cols])

# Exiba o DataFrame com os valores normalizados
print(dados_escola)

###1.7. Transforma valores em númericos
copia_dados_escola = dados_escola.copy()
targets = ['Abaixo da média','Média','Acima da média']
map_to_int = {name: n for n, name in enumerate(targets)}
copia_dados_escola["Classe-IDEB"] = copia_dados_escola["Classe-IDEB"].replace(map_to_int)
print(targets)

# Exibe o DataFrame atualizado
dados_escola = copia_dados_escola
print(dados_escola)

###1.9. Salva a Base
dados_escola.to_csv("Base de Dados/Base Pre Processada/base_tratada.csv")

##2. Criação do Modelo

####2.1 Instador das Bibliotecas
#Link PYcaret https://pycaret.gitbook.io/docs/get-started/quickstart#classification

####2.2 Importa Biblioteca
from pycaret.classification import *
from pycaret import classification

####2.3 Carrega a base
dados_escola = pd.read_csv("Base de Dados/Base Pre Processada/base_tratada.csv")

####2.4 Remove Coluna "Unnamed: 0"
if "Unnamed: 0" in dados_escola.columns:
    dados_escola.drop(columns=["Unnamed: 0"], inplace=True)

####2.5 Configura o ambiente
clf = setup(data=dados_escola, target='Classe-IDEB', session_id=1)

####2.6 Compara os modelos
best_model = compare_models()
print(best_model)

####2.9 Criando o Modelo (Extra Trees Classifier)
classification_et = classification.create_model('et')

classification.plot_model(classification_et, plot = 'feature', scale= 1)

classification.evaluate_model(classification_et)



