# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fase 3 - Balanceamento dos Dados
#
# Este arquivo implementa o balanceamento das classes da variável-alvo para evitar que o modelo de aprendizado de máquina se torne enviesado para a classe majoritária. Para isso, foi utilizado o SMOTE (Synthetic Minority Over-sampling Technique), uma técnica que cria amostras sintéticas da classe minoritária, equilibrando assim a distribuição das classes no conjunto de dados.
#
# ## Etapas do Balanceamento dos Dados
#
# ### 1. **Carregamento dos Dados**  
# O conjunto de dados de treino é carregado a partir do arquivo salvo na fase anterior, possibilitando a continuação do fluxo de preparação.
#
# ### 2. **Aplicação do SMOTE para Balanceamento das Classes**
# Foi utilizado o SMOTE para gerar amostras sintéticas da classe minoritária no conjunto de treino. Essa técnica ajuda a balancear as classes, evitando que o modelo aprenda de forma enviesada para a classe mais frequente
#
# ### 3. **Verificação das Proporções das Classes**
# Após a aplicação do SMOTE, foi exibido as novas proporções das classes no conjunto de dados de treino para confirmar o balanceamento
#
# ### 4. Salvamento dos Dados Balanceados
# Por fim, os dados balanceados são salvos em um arquivo para serem utilizados na próxima fase de treinamento e teste dos modelos.

# %%
import pandas as pd
from imblearn.over_sampling import SMOTE
import joblib

# %%
# Carregar os dados de treino e teste
X_train, X_test, y_train, y_test = joblib.load('data/data_splits.pkl')

# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Exibir as novas proporções
print("Proporções após SMOTE:")
print(y_resampled.value_counts(normalize=True))

# Salvar os dados reamostrados
joblib.dump((X_resampled, y_resampled), 'data/data_resampled.pkl')
