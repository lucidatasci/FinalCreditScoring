import pandas as pd
import numpy as np
import streamlit as st
import pickle
from pycaret.classification import load_model, predict_model
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

# Funções de conversão de DataFrame
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def substituir_nulos(df):
    df.dropna(inplace=True)
    return df

def remover_outliers(df):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
    return df

def criar_dummies(df):
    return pd.get_dummies(df, drop_first=True)

def selecionar_variaveis(X, y, n_features=8):
    model = RandomForestClassifier()
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-n_features:]  # Seleciona os índices das n_features mais importantes
    selected_columns = X.columns[indices]  # Nomes das variáveis selecionadas
    print("Variáveis selecionadas:", selected_columns)
    return X[selected_columns]

def aplicar_pca(X, n_components=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])

def preprocessamento(df):
    df = substituir_nulos(df)
    df = remover_outliers(df)
    y = df['mau']
    X = df.drop(columns=['mau'])
    X = criar_dummies(X)    
    X_selecionado = selecionar_variaveis(X, y, n_features=8)
    X_pca = aplicar_pca(X_selecionado, n_components=5)
    X_pca['mau'] = y.reset_index(drop=True)
    return X_pca

def preprocessamento_transformer(df):
    return preprocessamento(df)

def main():
    st.set_page_config(page_title='PyCaret', layout="wide", initial_sidebar_state='expanded')

    st.write("""## Credit Scoring com Pycaret """)
    st.write("Faça o upload dos dados e escolha o modelo a ser ajustado ao lado.")
    st.markdown("---")

    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Bank Credit Dataset", type=['csv', 'ftr'])

    model_choice = st.sidebar.selectbox("Escolha o modelo", ['LightGBM', 'Regressão Logística'])
    start_analysis = st.sidebar.button("Analyze!")

    # Criação da pipeline
    preprocessamento_pipeline = Pipeline([
        ('preprocessamento', FunctionTransformer(preprocessamento_transformer, validate=False)),
    ])

    if data_file_1 is not None:
        df_credit = pd.read_feather(data_file_1)
        df_credit.drop(columns=['data_ref', 'index'], inplace=True)
        df_credit = df_credit.sample(50000)
        X = df_credit.drop(columns='mau')
        
        st.write("Dados originais:")
        st.write(df_credit.head())

        # Carregar o pré-processamento e o modelo
        if model_choice == 'LightGBM':
            model_saved = load_model('Final LightGBM Model')
        elif model_choice == 'Regressão Logística':
            model_saved = pickle.load(open('model_final.pkl', 'rb'))

        if start_analysis:
            # Aplicar o pré-processamento
            try:
                df_transformed = preprocessamento_pipeline.transform(df_credit)
                X_transformed = df_transformed.drop(columns='mau')
                st.write("Dados transformados:", df_transformed.head())
                if model_choice == 'LightGBM':
                    predict = predict_model(model_saved, data=X)
                    st.write("Previsões:", predict.head())
                    df_xlsx = to_excel(predict)
                elif model_choice == 'Regressão Logística':
                    y_pred = model_saved.predict(X_transformed)
                    idx = X_transformed.index
                    predict = pd.DataFrame({
                        'Prediction': y_pred,
                        'Index': idx
                    }).set_index('Index')
                    df_credit['Prediction'] = df_credit.index.map(predict['Prediction'])
                    st.write("Previsões:", df_credit.head())
                    df_xlsx = to_excel(df_credit)



                
                st.download_button(label='📥 Download', data=df_xlsx, file_name='predict.xlsx')
            except Exception as e:
                st.error(f"Erro ao processar os dados: {e}")

if __name__ == '__main__':
    main()










