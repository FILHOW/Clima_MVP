# core/database.py (usando SQLite)

import sqlite3
import pandas as pd
import os
from sqlalchemy import create_engine

# Nome do arquivo do banco de dados SQLite
DB_NAME = "weather.db"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(PROJECT_ROOT, DB_NAME)
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "Summary of Weather.csv")

def create_database():
    """Cria um novo arquivo de banco de dados SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.close()
        print(f"Banco de dados SQLite '{DB_NAME}' criado com sucesso.")
    except sqlite3.Error as err:
        print(f"Erro ao criar banco de dados SQLite: {err}")

def create_tables():
    """Cria todas as tabelas (SOR, SOT, SPEC) no banco de dados SQLite."""
    tables_sql = {
        'sor_summary_of_weather': """
            CREATE TABLE IF NOT EXISTS sor_summary_of_weather (
                id INTEGER PRIMARY KEY NOT NULL,
                Date DATE NOT NULL,
                MaxTemp REAL NOT NULL,
                MinTemp REAL NOT NULL,
                MeanTemp REAL NOT NULL,
                YR INTEGER NOT NULL
            )
        """,
        'sot_summary_of_weather': """
            CREATE TABLE IF NOT EXISTS sot_summary_of_weather (
                numero_id INTEGER PRIMARY KEY NOT NULL,
                valor_data DATE NOT NULL,
                numero_tempMaxima REAL NOT NULL,
                numero_tempMinima REAL NOT NULL,
                numero_tempMedia REAL NOT NULL,
                numero_ano INTEGER NOT NULL
            )
        """,
        'spec_summary_of_weather': """
            CREATE TABLE IF NOT EXISTS spec_summary_of_weather (
                id INTEGER PRIMARY KEY NOT NULL,
                valor_data DATE NOT NULL,
                numero_tempMaxima REAL NOT NULL,
                numero_tempMinima REAL NOT NULL,
                numero_tempMedia REAL NOT NULL,
                numero_ano_42 BOOLEAN,
                numero_ano_43 BOOLEAN,
                numero_ano_44 BOOLEAN,
                numero_ano_45 BOOLEAN
            )
        """
    }
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for table_name, sql_script in tables_sql.items():
            try:
                cursor.execute(sql_script)
                print(f"Tabela '{table_name}' criada com sucesso.")
            except sqlite3.Error as err:
                print(f"Erro ao criar a tabela '{table_name}': {err}")
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
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV para SOR: {e}")
        return
    try:
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
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV para SOT: {e}")
        return
    try:
        engine = create_engine(f"sqlite:///{DB_PATH}")
        df_filtered.to_sql('sot_summary_of_weather', con=engine, if_exists='append', index=False)
        print("Dados inseridos com sucesso na tabela 'sot_summary_of_weather'.")
    except Exception as e:
        print(f"Erro ao inserir dados no SQLite para SOT: {e}")

def insert_spec_data():
    """Insere dados do CSV na tabela 'spec_summary_of_weather', com colunas dummy para o ano."""
    try:
        df = pd.read_csv(CSV_PATH, sep=',', low_memory=False)
        df['id'] = range(1, len(df) + 1)
        df['YR'] = df['YR'].astype(str)
        df_dummies = pd.get_dummies(df, columns=['YR'], prefix='numero_ano')
        df_dummies = df_dummies.rename(columns={
            'Date': 'valor_data',
            'MaxTemp': 'numero_tempMaxima',
            'MinTemp': 'numero_tempMinima',
            'MeanTemp': 'numero_tempMedia'
        })
        cols_to_keep = ['id', 'valor_data', 'numero_tempMaxima', 'numero_tempMinima', 'numero_tempMedia', 'numero_ano_42', 'numero_ano_43', 'numero_ano_44', 'numero_ano_45']
        df_filtered = df_dummies[cols_to_keep]
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV para SPEC: {e}")
        return
    try:
        engine = create_engine(f"sqlite:///{DB_PATH}")
        df_filtered.to_sql('spec_summary_of_weather', con=engine, if_exists='append', index=False)
        print("Dados inseridos com sucesso na tabela 'spec_summary_of_weather'.")
    except Exception as e:
        print(f"Erro ao inserir dados no SQLite para SPEC: {e}")

def drop_database():
    """Deleta o arquivo do banco de dados SQLite."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Banco de dados '{DB_NAME}' deletado com sucesso.")
    else:
        print(f"Banco de dados '{DB_NAME}' não encontrado.")