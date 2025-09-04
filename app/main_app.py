# app/main_app.py

import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, List

from core import database

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Análise de Clima — Fundamentos de IA", page_icon="🌦️", layout="wide")
st.title("🌦️ Análise de Dados Climáticos — Fundamentos de IA")

with st.sidebar:
    st.header("⚙️ Configuração da Aplicação")
    up = st.file_uploader("1. Faça upload do CSV (ex.: Summary of Weather.csv)", type=["csv"])
    alvo = st.selectbox("2. Alvo (target)", ["MaxTemp", "MinTemp", "MeanTemp"], index=0)
    pergunta = st.text_input("3. Pergunta (ex.: quais fatores afetam a temperatura?)",
                             value="Quais fatores afetam a temperatura?")
    test_size = st.slider("4. Proporção de teste", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("5. random_state", value=42, step=1)
    btn = st.button("🚀 Rodar Análise")
    
    st.markdown("---")
    st.header("🗄️ Opções de Banco de Dados (SQLite)")
    btn_create_db = st.button("1. Criar Banco de Dados")
    btn_create_tables = st.button("2. Criar Tabelas (SOR, SOT, SPEC)")
    btn_insert_sor = st.button("3. Inserir Dados (SOR)")
    btn_insert_sot = st.button("4. Inserir Dados (SOT)")
    btn_insert_spec = st.button("5. Inserir Dados (SPEC)")
    btn_drop_db = st.button("6. Dropar o Banco de Dados") # Botão para dropar o banco de dados
    
    st.markdown("---")
    st.header("💾 Salvar o Modelo")
    btn_save_model = st.button("7. Salvar Modelo Treinado")


# Lógica de processamento e treinamento do modelo (SEM ALTERAÇÕES)
st.markdown("""
**O que este app faz?**
1. Você envia a base (`Summary of Weather.csv`).
2. Escolhe o alvo (`MaxTemp`, `MinTemp` ou `MeanTemp`).
3. Faz sua pergunta.
4. O app treina um modelo de regressão linear e responde com os **fatores mais relevantes** e **métricas**.
""")

# ---------- Helpers de compatibilidade e pré-processamento ----------
def make_ohe():
    """Cria OneHotEncoder compatível com sklearn novo e antigo."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_processor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    categorical_processor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_processor, numeric_cols),
            ("cat", categorical_processor, categorical_cols),
        ],
        remainder="drop"
    )
    return preprocessor, numeric_cols, categorical_cols

# ---------- Importância/explicação (Regressão Linear) ----------
def regression_importance_from_linear(model: Pipeline, X_train: pd.DataFrame) -> pd.DataFrame:
    preproc = model.named_steps["preprocessor"]
    reg = model.named_steps["regressor"]

    try:
        feat_names = preproc.get_feature_names_out()
    except Exception:
        try:
            num_cols = preproc.transformers_[0][2]
        except Exception:
            num_cols = []
        try:
            cat_cols = preproc.transformers_[1][2]
            ohe = preproc.named_transformers_["cat"].named_steps["onehot"]
            cat_feat = ohe.get_feature_names_out(cat_cols)
        except Exception:
            cat_feat = []
        feat_names = np.array(list(num_cols) + list(cat_feat))

    Xtr_trans = preproc.transform(X_train)
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

# Função para salvar o modelo
def save_model(model: Pipeline, model_path: str = 'model/model.pickle'):
    """Salva o modelo treinado em um arquivo .pickle."""
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        st.success(f"✅ Modelo salvo com sucesso em '{model_path}'")
    except Exception as e:
        st.error(f"Erro ao salvar o modelo: {e}")

# ========== Fluxo da UI ==========
if btn_create_db:
    database.create_database()
    st.info("Função `create_database()` executada. Verifique o terminal para o resultado.")
elif btn_create_tables:
    database.create_tables()
    st.info("Função `create_tables()` executada. Verifique o terminal para o resultado.")
elif btn_insert_sor:
    database.insert_sor_data()
    st.info(f"Função `insert_sor_data()` executada. Verifique o terminal para o resultado.")
elif btn_insert_sot:
    database.insert_sot_data()
    st.info(f"Função `insert_sot_data()` executada. Verifique o terminal para o resultado.")
elif btn_insert_spec:
    database.insert_spec_data()
    st.info(f"Função `insert_spec_data()` executada. Verifique o terminal para o resultado.")
elif btn_drop_db:
    database.drop_database()
    st.info("Função `drop_database()` executada. Verifique o terminal para o resultado.")

if not up and not btn and not btn_save_model:
    st.info("⬅️ Envie um CSV (`Summary of Weather.csv`) e/ou clique em **Rodar** após configurar.")
elif btn and not up:
    st.warning("⚠️ Você clicou em **Rodar**, mas nenhum CSV foi enviado. Anexe um arquivo primeiro.")
elif up:
    try:
        df = pd.read_csv(up, sep=',', low_memory=False)
        st.success("✅ Arquivo carregado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        st.stop()

    st.subheader("👀 Pré-visualização dos dados")
    st.dataframe(df.head(10), use_container_width=True)

    cols_sug = ["MaxTemp", "MinTemp", "MeanTemp", "YR", "MO"]
    disp_cols = [c for c in cols_sug if c in df.columns]
    if len(disp_cols) >= 3:
        st.caption("Sugestão de colunas (Clima padrão): " + ", ".join(disp_cols))

    if btn:
        if alvo not in df.columns:
            st.error(f"O alvo '{alvo}' não existe no CSV.")
            st.stop()

        drop_candidates = [alvo, "Date", "Station", "id"]
        X_full = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")
        y_full = df[alvo]
        y_full = pd.to_numeric(y_full, errors='coerce')

        mask_notna = y_full.notna()
        X_full, y_full = X_full[mask_notna], y_full[mask_notna]

        preproc, _, _ = build_preprocessor(X_full)
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

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📈 Importância / Coeficientes")
            imp = regression_importance_from_linear(model, X_train)
            st.dataframe(imp[["Feature","Coef_Linear","Impacto_Relativo"]].head(20), use_container_width=True)

        with col2:
            st.markdown("### 📊 Métricas")
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            st.write(f"RMSE (teste): **{rmse:.4f}**")

        st.markdown("### 💬 Resposta do Chatbot")
        st.info(answer_from_question(pergunta, imp))
        st.success("Pronto! Você pode ajustar a pergunta, trocar o alvo e reenviar a base.")
    
    # Adicionando o botão para salvar o modelo
    if btn_save_model:
        if 'model' in locals():
            save_model(model)
        else:
            st.warning("⚠️ Você precisa rodar o modelo primeiro antes de salvá-lo. Clique em '🚀 Rodar Análise' para treinar o modelo.")
