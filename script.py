#BIBLIOTECAS UTILIZADAS
import os
import wget
import pandas as pd
import zipfile
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress

#EXTRAÇÃO DOS DADOS DA FONTE DE DADOS
if os.path.exists("CONTRATOS.zip"):
    os.remove("CONTRATOS.zip")

URL = "https://www.compras.rj.gov.br/siga/imagens/CONTRATOS.zip"

caminho_download = 'C:\\Users\\Gabriel\\Desktop\\PROJETO_ANALISE_DE_DADOS'

response = wget.download(URL, caminho_download)

caminho_zip = 'C:\\Users\\Gabriel\\Desktop\\PROJETO_ANALISE_DE_DADOS\\CONTRATOS.zip'

with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
    zip_ref.extractall('TABELAS_EXCEL_CSV')

#TRATAMENTO E TRANSFORMAÇÃO DE DADOS
tipos_de_dados = {
    #COLUNAS CONTRATOS
    'Contratação': int,
    'Status Contratação': object,
    'Unidade': object,
    'Processo': object,
    'Objeto': object,
    'Tipo de Aquisição': object,
    'Critério de Julgamento': object,
    'Fornecedor': object,
    'CPF/CNPJ': object,
    'Valor Total Contrato/Valor Estimado para Contratação (R$)': float,
    'Valor Total Empenhado (R$)': float,
    'Valor Total Liquidado (R$)': float,
    'Valor Total Pago (R$)': float,
    #COLUNAS ITENS_CONTRATOS
    'ID Item': int,
    'Item': object,
    'Qtde Original': float,
    'VL. Unit.Original': float,
    'Total Aditivada/Suprimida': float,
    'VL. Unit.Aditivado/Suprimido': float,
}

dados_numericos = [
    'Valor Total Contrato/Valor Estimado para Contratação (R$)',
    'Valor Total Empenhado (R$)',
    'Valor Total Liquidado (R$)',
    'Valor Total Pago (R$)',
    'Qtde Original',
    'VL. Unit.Original',
    'Total Aditivada/Suprimida',
    'VL. Unit.Aditivado/Suprimido'
]

column_options = {
    'Unidade': {'max_colwidth': 50},
    'Processo': {'max_colwidth': 50},
    'Objeto': {'max_colwidth': 50},
    'Tipo de Aquisição': {'max_colwidth': 50},
    'Fornecedor': {'max_colwidth': 50},
    'Item': {'max_colwidth': 50},
}

df_contratos = pd.read_csv('TABELAS_EXCEL_CSV/CONTRATOS.csv', sep=';', encoding='latin1', dtype=tipos_de_dados, decimal=',', thousands='.')
df_itens_contratos = pd.read_csv('TABELAS_EXCEL_CSV/ITENS_CONTRATOS.csv', sep=';', encoding='latin1', dtype=tipos_de_dados, decimal=',', thousands='.')
del df_contratos['data_extracao']
df_merged = pd.merge(df_contratos, df_itens_contratos, on='Contratação')


df_merged['Data Contratação'] = pd.to_datetime(df_merged['Data Contratação'], errors='coerce', dayfirst=True).dt.strftime('%d/%m/%Y')
df_merged['Data Início Vigência'] = pd.to_datetime(df_merged['Data Início Vigência'], errors='coerce', dayfirst=True).dt.strftime('%d/%m/%Y')
df_merged['Data Fim Vigência'] = pd.to_datetime(df_merged['Data Fim Vigência'], errors='coerce', dayfirst=True).dt.strftime('%d/%m/%Y')
df_merged['Data Public DEORJ'] = pd.to_datetime(df_merged['Data Public DEORJ'], errors='coerce', dayfirst=True).dt.strftime('%d/%m/%Y')
df_merged['data_extracao'] = pd.to_datetime(df_merged['data_extracao'], errors='coerce', dayfirst=True).dt.strftime('%d/%m/%Y')


pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
for col, options in column_options.items():
    pd.set_option(f"display.max_colwidth", options['max_colwidth'])


    
dados_numericos = [col.strip() for col in dados_numericos] # Remover espaços em branco dos nomes das colunas em dados_numericos
df_merged.columns = df_merged.columns.str.strip() # Remover espaços em branco do índice do df_merged

# Inicialize o objeto SimpleImputer para preencher os valores ausentes com a média
imputer = SimpleImputer(strategy='mean')

# Ajuste o imputer aos seus dados originais
imputer.fit(df_merged[dados_numericos])

# Preencha os valores ausentes com a média
df_merged[dados_numericos] = imputer.transform(df_merged[dados_numericos])

# Inicialize o objeto PCA com o número desejado de componentes
num_componentes = 3  # Escolha o número de componentes desejado
pca = PCA(n_components=num_componentes)

# Ajuste o modelo PCA aos seus dados originais
pca.fit(df_merged[dados_numericos])

# Transforme os seus dados originais no novo espaço de características definido pelas componentes principais
dados_transformados = pca.transform(df_merged[dados_numericos])

# Instanciar o objeto PCA
pca = PCA()

# Aplicar o PCA aos dados
pca.fit(df_merged[dados_numericos])

# Reduzir a dimensionalidade dos dados
df_reduced = pca.transform(df_merged[dados_numericos])

# Calcular a variância explicada acumulada
explained_variance_ratio_cumulative = pca.explained_variance_ratio_.cumsum()

#CARGA

# Plotar a variância explicada acumulada
plt.plot(range(1, len(explained_variance_ratio_cumulative) + 1), explained_variance_ratio_cumulative, marker='o')
plt.xlabel('Número de componentes principais')
plt.ylabel('Variância explicada acumulada')
plt.title('Variância explicada acumulada pelos componentes principais')
plt.grid(True)
plt.show()

# Visualizar os dados reduzidos em um gráfico de dispersão bidimensional
plt.scatter(df_reduced[:, 0], df_reduced[:, 2])
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Dados reduzidos pelos componentes principais')
plt.grid(True)
plt.show()

#CORRELAÇÃO DE PEARSON
# Calcula a matriz de correlação de Pearson
correlation_matrix = df_merged[dados_numericos].corr()

# Plota o mapa de calor da matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação de Pearson')
plt.show()

#GRAFICO DE MÉDIA
# Extrai o ano da coluna 'Data Contratação'
df_merged['Data Contratação'] = pd.to_datetime(df_merged['Data Contratação'], dayfirst=True, errors='coerce').dt.strftime('%d/%m/%Y')
df_merged['Ano Contratação'] = pd.to_datetime(df_merged['Data Contratação'], dayfirst=True, errors='coerce').dt.year

# Filtra o DataFrame para incluir apenas os anos de 2016 a 2023
df_filtered = df_merged[(df_merged['Ano Contratação'] >= 2016) & (df_merged['Ano Contratação'] <= 2023)]

# Calcula a média do Valor Total Pago por ano
media_valor_pago_por_ano = df_filtered.groupby('Ano Contratação')['Valor Total Pago (R$)'].mean()

# Configurações do gráfico
plt.figure(figsize=(10, 6))
sns.barplot(x=media_valor_pago_por_ano.index, y=media_valor_pago_por_ano.values, hue=media_valor_pago_por_ano.index, palette='viridis', legend=False)

# Adiciona título e rótulos aos eixos
plt.title('Média do Valor Total Pago por Ano de Contratação (2016-2023)')
plt.xlabel('Ano de Contratação')
plt.ylabel('Média do Valor Total Pago (R$)')

# Mostra o gráfico
plt.tight_layout()
plt.show()

#GRAFICO DE DISPERSÃO
# Calcula o desvio padrão do Valor Total Pago por ano
desvio_padrao_valor_pago_por_ano = df_filtered.groupby('Ano Contratação')['Valor Total Pago (R$)'].std()

# Calcula os coeficientes da regressão linear
slope, intercept, _, _, _ = linregress(desvio_padrao_valor_pago_por_ano.index, desvio_padrao_valor_pago_por_ano.values)

# Calcula os valores previstos para a linha de tendência diagonal
x_values = np.arange(2016, 2024)
predicted_y_values = slope * x_values + intercept

# Configurações do gráfico de dispersão
plt.figure(figsize=(10, 6))

# Gráfico de dispersão
sns.scatterplot(x=desvio_padrao_valor_pago_por_ano.index, y=desvio_padrao_valor_pago_por_ano.values)

# Adiciona a linha de tendência diagonal
plt.plot(x_values, predicted_y_values, color='red', linestyle='--', label='Linha de Tendência Diagonal')

# Adiciona título e rótulos aos eixos
plt.title('Dispersão do Desvio Padrão do Valor Total Pago por Ano de Contratação (2016-2023)')
plt.xlabel('Ano de Contratação')
plt.ylabel('Desvio Padrão do Valor Total Pago (R$)')

# Adiciona legenda
plt.legend()

# Mostra o gráfico
plt.tight_layout()
plt.show()