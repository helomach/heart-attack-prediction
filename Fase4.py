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
# # Fase 4 - Treinamento e Teste
#
# Este arquivo realiza o treinamento e a avaliação de dois modelos de classificação – Regressão Logística e Random Forest – utilizando dados de treino previamente balanceados com SMOTE. Abaixo, estão as principais etapas executadas no script:
#
# ### 1. **Carregamento dos Dados**  
# São carregados os dados de treino balanceados ```data_resampled.pkl``` e o conjunto de teste ```data_test.pkl```
#
# ### 2. **Treinamento dos Modelos**
# * **Regressão Logística:** O modelo de regressão logística é treinado com os dados balanceados ```X_resampled, y_resampled```, e o estado aleatório é fixado para 42 para garantir reprodutibilidade;
# * **Random Forest:** Um modelo de Random Forest é treinado usando os mesmos dados balanceados, também com o estado aleatório fixado.
#
# ### 3. **Avaliação dos Modelos**
# * **Previsões com o Conjunto de Teste:** Ambos os modelos são usados para prever os rótulos do conjunto de teste ```X_test```. Os resultados são armazenados em ```y_pred_log``` para Regressão Logística e ```y_pred_rf``` para Random Forest;
# * **Exibição dos Resultados:** Para cada modelo, é exibido o relatório de classificação (com métricas como acurácia, precisão, recall, e F1-score) e a matriz de confusão, facilitando a análise de desempenho.
#
# ### 4. **Salvar os Modelos Treinados**
# * **Persistência dos Modelos:** Os modelos treinados são salvos para uso posterior. Eles são armazenados como ```log_model.pkl``` para o modelo de Regressão Logística e ```rf_model.pkl``` para o modelo de Random Forest.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# %%
# Carregar os dados de treino balanceados
X_resampled, y_resampled = joblib.load('data/data_resampled.pkl')

# Carregar os dados de teste
X_test, y_test = joblib.load('data/data_test.pkl')

# %%
# Treinamento com Regressão Logística
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_resampled, y_resampled)

# Treinamento com Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)

# %%
# Avaliação com o conjunto de teste
y_pred_log = log_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# %%
# Exibir resultados
print("Resultados da Regressão Logística:")
print(classification_report(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))

print("\nResultados da Random Forest:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# %%
# Salvar os modelos treinados
joblib.dump(log_model, 'data/log_model.pkl')
joblib.dump(rf_model, 'data/rf_model.pkl')
