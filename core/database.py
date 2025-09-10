import sqlite3
import pandas as pd
import os
from sqlalchemy import create_engine
import glob

# Nome do arquivo do banco de dados SQLite
DB_NAME = "weather.db"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(PROJECT_ROOT, DB_NAME)
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "Summary of Weather.csv")
SQL_PATH = os.path.join(PROJECT_ROOT, "sql", "*.sql")

def create_database():
    """Apaga o banco de dados se ele existir e cria um novo."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Banco de dados '{DB_NAME}' apagado com sucesso.")

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.close()
        print(f"Banco de dados SQLite '{DB_NAME}' criado com sucesso.")
    except sqlite3.Error as err:
        print(f"Erro ao criar o banco de dados SQLite: {err}")

def create_tables_from_sql():
    """Lê os arquivos .sql e cria as tabelas no banco de dados."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        sql_files = glob.glob(SQL_PATH)
        if not sql_files:
            print("Nenhum arquivo .sql encontrado na pasta 'sql'.")
            return
            
        for sql_file in sql_files:
            with open(sql_file, 'r') as f:
                sql_script = f.read()
                try:
                    cursor.executescript(sql_script)
                    print(f"Tabelas criadas a partir do arquivo '{os.path.basename(sql_file)}'.")
                except sqlite3.Error as err:
                    print(f"Erro ao executar o script SQL do arquivo '{sql_file}': {err}")
        conn.close()
    except sqlite3.Error as err:
        print(f"Erro ao conectar ao banco de dados: {err}")

def insert_sor_data():
    """Insere dados do CSV na tabela 'sor_summary_of_weather'."""
    try:
        df = pd.read_csv(CSV_PATH, sep=',', low_memory=False)
        df['id'] = range(1, len(df) + 1)
        cols_to_keep = ['id', 'Date', 'MaxTemp', 'MinTemp', 'MeanTemp', 'YR']
        df_filtered = df[cols_to_keep]
        engine = create_engine(f"sqlite:///{DB_PATH}")
        df_filtered.to_sql('sor_summary_of_weather', con=engine, if_exists='append', index=False)
        print("Dados inseridos com sucesso na tabela 'sor_summary_of_weather'.")
    except Exception as e:
        print(f"Erro ao inserir dados no SQLite para SOR: {e}")

def insert_sot_data():
    """Insere dados do CSV na tabela 'sot_summary_of_weather', renomeando as colunas."""
    try:
        df = pd.read_csv(CSV_PATH, sep=',', low_memory=False)
        df['numero_id'] = range(1, len(df) + 1)
        df = df.rename(columns={
            'Date': 'valor_data',
            'MaxTemp': 'numero_tempMaxima',
            'MinTemp': 'numero_tempMinima',
            'MeanTemp': 'numero_tempMedia',
            'YR': 'numero_ano'
        })
        cols_to_keep = ['numero_id', 'valor_data', 'numero_tempMaxima', 'numero_tempMinima', 'numero_tempMedia', 'numero_ano']
        df_filtered = df[cols_to_keep]
        engine = create_engine(f"sqlite:///{DB_PATH}")
        df_filtered.to_sql('sot_summary_of_weather', con=engine, if_exists='append', index=False)
        print("Dados inseridos com sucesso na tabela 'sot_summary_of_weather'.")
    except Exception as e:
        print(f"Erro ao inserir dados no SQLite para SOT: {e}")

def insert_spec_data():
    """
    Insere dados do CSV na tabela 'spec_summary_of_weather',
    incluindo apenas as colunas de features que o modelo precisa.
    """
    try:
        df = pd.read_csv(CSV_PATH, sep=',', low_memory=False)
        
        # Adiciona a coluna 'id'
        df['id'] = range(1, len(df) + 1)
        
        # Cria as colunas one-hot encoded para os anos
        df = pd.get_dummies(df, columns=['YR'], prefix='YR')
        
        # Filtra as colunas desejadas para o modelo e para o banco
        features = ['id', 'MaxTemp', 'MinTemp', 'YR_40', 'YR_41', 'YR_42', 'YR_43', 'YR_44', 'YR_45']
        
        # Garante que todas as colunas existem no DataFrame antes de selecionar
        for col in features:
            if col not in df.columns:
                df[col] = 0 # Adiciona a coluna com valor 0 se ela não existir
        
        df_filtered = df[features]
        
        engine = create_engine(f"sqlite:///{DB_PATH}")
        df_filtered.to_sql('spec_summary_of_weather', con=engine, if_exists='append', index=False)
        print("Dados inseridos com sucesso na tabela 'spec_summary_of_weather'.")
    except Exception as e:
        print(f"Erro ao inserir dados no SQLite para SPEC: {e}")

def get_spec_data_from_db():
    """Lê dados da tabela SPEC no banco de dados."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM spec_summary_of_weather", conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        raise Exception(f"Erro ao ler dados do banco de dados: {e}")
