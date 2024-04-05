import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

prod_file = '../data/processed/prediction_prod.parquet'
dev_file = '../data/processed/prediction_test.parquet'

############################################ SIDE BAR TITLE
st.sidebar.title('Painel de Controle')
st.sidebar.markdown(f"""
Controle do tipo de vinho analisado e entrada de variáveis para avaliação de novos vinhos.
""")


df_prod = pd.read_parquet(prod_file)
df_dev = pd.read_parquet(dev_file)

# st.write(df_prod)
# st.write(df_dev)


fignum = plt.figure(figsize=(6,4))
# Saida do modelo dados dev
sns.distplot(df_dev.prediction_score_1,
             label='Teste',
             ax = plt.gca())

# Saida do modelo dados prod
sns.distplot(df_prod.predict_score,
             label='Producao',
             ax = plt.gca())



# User wine

plt.title('Monitoramento Desvio de Dados da Saída do Modelo')
plt.ylabel('Densidade Estimada')
plt.xlabel('Probabilidade Vinho Alta Qualidade')
plt.xlim((0,1))
plt.grid(True)
plt.legend(loc='best')

st.pyplot(fignum)



from sklearn import metrics

st.write(metrics.classification_report(df_dev.target, df_dev.prediction_label))





