# Arquitetura de Software — Kaggle Chatbot MVP

> Este documento descreve a arquitetura lógica, a organização em camadas e o fluxo de dados do MVP, além de instruções detalhadas para desenhar o diagrama no **draw.io** (diagrams.net) e exportar para o repositório.

## 1. Objetivos de Arquitetura

- **Separação de responsabilidades**: UI (Streamlit) isolada da lógica de negócio (`core/`) e dos artefatos (`data/`).
- **Simplicidade (MVP)**: poucos componentes, acoplamento baixo e convenções claras.
- **Reprodutibilidade**: pipelines de ML encapsulados; execução determinística.
- **Evolução**: fácil trocar o tema Kaggle, o modelo e incluir IA generativa.
- **Governança**: espaço próprio para configs (`configs/`) e documentação (`docs/`).

## 2. Visão em Camadas

```
Usuário (navegador)
   ↓
[ app/ ]  → Streamlit UI
   ↓
[ core/ ]
   ├─ data/       → leitura/validação de CSV
   ├─ features/   → pipelines de preprocessamento
   ├─ models/     → treino, avaliação, predição
   ├─ explain/    → coeficientes/importâncias
   └─ chatbot/    → regras de resposta
   ↓
[ data/ ]
   ├─ raw/        → dados brutos
   ├─ processed/  → dados tratados
   └─ models/     → artefatos .pkl
```

## 3. Componentes e Responsabilidades

### app/
- Camada de apresentação (UI com Streamlit).
- Upload de CSV, escolha de tarefa (classificação ou regressão).
- Exibição de métricas, importâncias e respostas do chatbot.

### core/data/
- Funções de I/O para CSVs.
- Validação de schema (colunas obrigatórias, tipos).

### core/features/
- Pipelines de preprocessamento: imputação, one-hot, escala.

### core/models/
- Treinamento de modelos (LogisticRegression, LinearRegression).
- Avaliação: métricas de classificação (acurácia, f1, precisão, recall) e regressão (RMSE).

### core/explain/
- Extração de coeficientes e odds ratios.
- Interpretação simples de variáveis mais relevantes.

### core/chatbot/
- Regras para perguntas frequentes (FAQ).
- Respostas sobre métricas, variáveis e pipeline.

### data/
- `raw/`: datasets brutos (não alterar).
- `processed/`: datasets tratados (quando necessário).
- `models/`: modelos salvos (.pkl).

### configs/
- Arquivos de configuração e logging.

### docs/
- Documentação integrada: PMC, arquitetura, dados, LGPD, testes, deploy.

## 4. Fluxo de Execução

1. Usuário acessa aplicação pelo navegador.
2. Faz upload do dataset (ex.: Summary of weather `Summary of weather.csv`).
3. A camada `app/` passa os dados para `core/data/` → validação.
4. `core/features/` monta pipeline de preprocessamento.
5. `core/models/` treina e avalia modelo escolhido.
6. `core/explain/` gera coeficientes/importâncias.
7. `core/chatbot/` interpreta perguntas e gera respostas.
8. Resultados (métricas, gráficos, texto) são exibidos em `app/`.
9. Artefatos (datasets tratados, modelos) são salvos em `data/`.

