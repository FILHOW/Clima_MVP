import sys
import os
# Adiciona o diretório raiz do projeto ao PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, List
from sqlalchemy import create_engine
from core import database
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Análise de Clima — Fundamentos de IA", page_icon="🌦️", layout="wide")
st.title("🌦️ Análise de Dados Climáticos — Fundamentos de IA")

# Inicializa o session_state para armazenar o modelo
if 'model' not in st.session_state:
    st.session_state.model = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuração da Aplicação")
    
    app_mode = st.radio("Escolha o modo de operação", ["Treinar Modelo", "Carregar Modelo Salvo"])
    st.session_state.app_mode = app_mode

    st.markdown("---")
    
    if st.session_state.app_mode == "Treinar Modelo":
        st.header("Treinamento do Modelo")
        up = st.file_uploader("1. Faça upload do CSV (ex.: Summary of Weather.csv)", type=["csv"])
        alvo = st.selectbox("2. Alvo (target)", ["MaxTemp", "MinTemp"], index=0)
        pergunta = st.text_input("3. Pergunta (ex.: quais fatores afetam a temperatura?)",
                                 value="Quais fatores afetam a temperatura?")
        test_size = st.slider("4. Proporção de teste", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("5. random_state", value=42, step=1)
        btn_run_analysis = st.button("🚀 Rodar Análise")
        btn_save_model = st.button("7. Salvar Modelo Treinado")

    elif st.session_state.app_mode == "Carregar Modelo Salvo":
        st.header("Predição com Modelo Salvo")
        btn_load_model = st.button("💾 Carregar Modelo")
        btn_predict = st.button("🔮 Rodar Predição")
        
    st.markdown("---")
    st.header("🗄️ Opções de Banco de Dados (SQLite)")
    btn_create_db = st.button("1. Criar Banco de Dados (Drop/Create)")
    btn_create_tables = st.button("2. Criar Tabelas (SOR, SOT, SPEC)")
    btn_insert_sor = st.button("3. Inserir Dados (SOR)")
    btn_insert_sot = st.button("4. Inserir Dados (SOT)")
    btn_insert_spec = st.button("5. Inserir Dados (SPEC)")
    
# Lógica de processamento e treinamento do modelo
st.markdown("""
**O que este app faz?**
1. Você envia a base (`Summary of Weather.csv`).
2. Escolhe o alvo (`MaxTemp` ou `MinTemp`).
3. Faz sua pergunta.
4. O app treina um modelo de regressão linear e responde com os **fatores mais relevantes** e **métricas**.
""")

# ---------- Helpers de compatibilidade e pré-processamento ----------
def build_preprocessor(X: pd.DataFrame) -> Pipeline:
    numeric_processor = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    return numeric_processor

# ---------- Importância/explicação (Regressão Linear) ----------
def regression_importance_from_linear(model: Pipeline, X_train: pd.DataFrame) -> pd.DataFrame:
    preproc = model.named_steps["preprocessor"]
    reg = model.named_steps["regressor"]

    feat_names = X_train.columns.tolist()
    
    Xtr_trans = preproc.fit_transform(X_train)
    std_x = Xtr_trans.std(axis=0)

    coefs = reg.coef_
    impacto = np.abs(coefs) * std_x

    imp = pd.DataFrame({"Feature": feat_names, "Coef_Linear": coefs, "Impacto_Relativo": impacto})
    imp["AbsCoef"] = np.abs(coefs)
    return imp.sort_values("Impacto_Relativo", ascending=False)

def answer_from_question(pergunta: str, tabela_imp: pd.DataFrame) -> str:
    p = (pergunta or "").lower()
    if any(k in p for k in ["característica","caracteristicas","feature","variável","variaveis","importante","importância","importancia"]):
        top_imp = tabela_imp[["Feature","Coef_Linear","Impacto_Relativo"]].head(5)
        return (
            "🔎 **Fatores mais associados à temperatura (Regressão)**\n\n"
            f"Top 5 por impacto relativo:\n{top_imp.to_markdown(index=False)}"
        )
    return "Posso analisar a **importância das variáveis** e **métricas**. Tente: *quais fatores são mais importantes?*"

# Function to save the model
def save_model(model: Pipeline, model_path: str = 'models/model.pickle'):
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        st.success(f"✅ Modelo salvo com sucesso em '{model_path}'")
    except Exception as e:
        st.error(f"Erro ao salvar o modelo: {e}")

# Function to load the model
def load_model(model_path: str = 'models/model.pickle'):
    if not os.path.exists(model_path):
        st.error(f"❌ O arquivo do modelo '{model_path}' não foi encontrado.")
        return None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success(f"✅ Modelo carregado com sucesso!")
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# ========== UI Flow ==========
if btn_create_db:
    database.create_database()
    st.info("Função `create_database()` executada.")
elif btn_create_tables:
    database.create_tables_from_sql()
    st.info("Função `create_tables_from_sql()` executada.")
elif btn_insert_sor:
    database.insert_sor_data()
    st.info(f"Função `insert_sor_data()` executada.")
elif btn_insert_sot:
    database.insert_sot_data()
    st.info(f"Função `insert_sot_data()` executada.")
elif btn_insert_spec:
    database.insert_spec_data()
    st.info(f"Função `insert_spec_data()` executada.")

if st.session_state.app_mode == "Treinar Modelo":
    if btn_save_model:
        if st.session_state.model:
            save_model(st.session_state.model)
        else:
            st.warning("⚠️ Você precisa rodar o modelo primeiro antes de salvá-lo. Clique em '🚀 Rodar Análise' para treinar o modelo.")

    if not up and not btn_run_analysis:
        st.info("⬅️ Envie um CSV (`Summary of Weather.csv`) e/ou clique em **Rodar** após configurar.")
    elif btn_run_analysis and not up:
        st.warning("⚠️ Você clicou em **Rodar**, mas nenhum CSV foi enviado. Anexe um arquivo primeiro.")
    elif up:
        try:
            df = pd.read_csv(up, sep=',', low_memory=False)
            st.success("✅ Arquivo carregado com sucesso!")
            
            # Realiza o get_dummies antes de treinar
            df = pd.get_dummies(df, columns=['YR'], prefix='YR')
            
            # Define as colunas de features e o alvo
            # As colunas de ano devem ser tratadas para garantir que todas existam
            years_to_predict = [40, 41, 42, 43, 44, 45]
            feature_cols = [f'YR_{year}' for year in years_to_predict]
            
            # Garante que todas as colunas de ano existem no DataFrame antes de selecionar
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0
            
            # Define as features e o alvo
            X_full = df[feature_cols]
            y_full = df[alvo]
            
            # Converte o tipo de dado para numérico, se necessário
            y_full = pd.to_numeric(y_full, errors='coerce')
            
            # Remove linhas com valores nulos no alvo
            mask_notna = y_full.notna()
            X_full, y_full = X_full[mask_notna], y_full[mask_notna]

            preproc = build_preprocessor(X_full)
            model = Pipeline(steps=[
                ("preprocessor", preproc),
                ("regressor", LinearRegression())
            ])

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_full, y_full, test_size=float(test_size), random_state=int(random_state)
                )
            except ValueError as e:
                st.error(f"Erro ao separar treino/teste: {e}")
                st.stop()

            model.fit(X_train, y_train)
            st.session_state.model = model

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📈 Importância / Coeficientes")
                imp = regression_importance_from_linear(st.session_state.model, X_train)
                st.dataframe(imp[["Feature","Coef_Linear","Impacto_Relativo"]], use_container_width=True)

            with col2:
                st.markdown("### 📊 Métricas")
                y_pred = st.session_state.model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                st.write(f"R² (teste): **{r2:.4f}**")
                st.write(f"MAE (teste): **{mae:.4f}**") # Adicionado MAE
                st.write(f"RMSE (teste): **{rmse:.4f}**")
            
            # --- Visualização de Predição vs. Real (adicionada) ---
            st.markdown("### 📈 Visualização de Predição")
            st.markdown("O gráfico abaixo compara as temperaturas previstas (eixo X) com as temperaturas reais (eixo Y).")
            
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Temperatura Real")
            ax.set_ylabel("Temperatura Prevista")
            ax.set_title("Temperatura Prevista vs. Temperatura Real")
            ax.grid(True)
            st.pyplot(fig)
            # --- Fim da visualização ---

            st.markdown("### 💬 Resposta do Chatbot")
            st.info(answer_from_question(pergunta, imp))

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo ou treinar o modelo: {e}")

elif st.session_state.app_mode == "Carregar Modelo Salvo":
    st.markdown("---")
    st.subheader("🔮 Predição com Modelo Carregado")
    
    if btn_load_model:
        st.session_state.model = load_model()
    
    if st.session_state.model:
        st.markdown("---")
        st.subheader("Predição de um ano específico")
        
        # Obtém os dados da tabela SPEC do banco de dados para a predição
        try:
            df_spec = database.get_spec_data_from_db()
            
            years_to_predict = [40, 41, 42, 43, 44, 45]
            
            # Cria um DataFrame para a predição
            df_predict = pd.DataFrame(columns=[f'YR_{year}' for year in years_to_predict])
            
            # Preenche o DataFrame com os dados de one-hot encoding
            for year in years_to_predict:
                df_predict[f'YR_{year}'] = (df_spec['YR_'+str(year)] > 0).astype(int)

            if btn_predict:
                # Faz a predição
                prediction = st.session_state.model.predict(df_predict)
                
                # Exibe o resultado
                df_predict['Predicted_MaxTemp'] = prediction
                
                # Exibe o DataFrame com a predição
                st.write("Predições do modelo para o ano selecionado:")
                st.dataframe(df_predict, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao ler dados da tabela SPEC ou ao fazer a predição: {e}")
