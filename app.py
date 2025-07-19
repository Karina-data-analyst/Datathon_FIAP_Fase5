import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import string
import unicodedata
import nltk
import gc
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack, csr_matrix

# --- Funções de Pré-processamento Otimizadas ---

@st.cache_data
def get_stopwords():
    """Carrega e retorna as listas de stopwords em português e inglês."""
    return set(stopwords.words('portuguese')), set(stopwords.words('english'))

def normalize_accents(text):
    if not isinstance(text, str): return ""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

def remove_punctuation(text):
    if not isinstance(text, str): return ""
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize_str(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = normalize_accents(text)
    text = remove_punctuation(text)
    text = re.sub(r" +", " ", text)
    return " ".join([w for w in text.split()])

@st.cache_resource
def download_nltk_resources():
    st.info("Verificando recursos NLTK (executado apenas uma vez)...")
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        st.error(f"Falha ao baixar recursos do NLTK: {e}")

# CORREÇÃO DA LÓGICA: A chamada para get_stopwords() foi movida para dentro do tokenizer.
def tokenizer(text):
    if not isinstance(text, str):
        return ""
    
    stop_words_pt, stop_words_eng = get_stopwords()
    
    text = normalize_str(text)
    text = "".join([w for w in text if not w.isdigit()])
    
    words = word_tokenize(text, language='portuguese')
    words = [x for x in words if x not in stop_words_pt and x not in stop_words_eng]
    words = [y for y in words if len(y) > 2]
    return " ".join(words)

# --- INÍCIO DAS FUNÇÕES FALTANTES ---

@st.cache_resource
def load_ml_resources():
    try:
        loaded_model = joblib.load(MODEL_FILEPATH)
        loaded_vectorizer = joblib.load(VECTORIZER_FILEPATH)
        return loaded_model, loaded_vectorizer
    except FileNotFoundError:
        st.error(f"Erro: Arquivos do modelo ('{MODEL_FILEPATH}') ou vetorizador ('{VECTORIZER_FILEPATH}') não encontrados.")
        return None, None

@st.cache_data
def calculate_match_scores_for_df(df, _model, _vectorizer):
    if df.empty or _model is None or _vectorizer is None:
        st.warning("Não foi possível calcular scores: DataFrame vazio ou recursos de ML não carregados.")
        if 'match_score' not in df.columns: df['match_score'] = np.nan
        if 'match_score_pct' not in df.columns: df['match_score_pct'] = np.nan
        return df

    st.info("Calculando Scores de Match...")
    
    mask_valid_text = (df['cv_tratados'].str.strip().astype(bool)) & \
                      (df['competencia_tecnicas_e_comportamentais_tratadas'].str.strip().astype(bool))
    
    df['match_score'] = np.nan

    if mask_valid_text.any():
        try:
            text_to_score_cv = df.loc[mask_valid_text, 'cv_tratados']
            text_to_score_job = df.loc[mask_valid_text, 'competencia_tecnicas_e_comportamentais_tratadas']

            cv_vectors = _vectorizer.transform(text_to_score_cv)
            job_desc_vectors = _vectorizer.transform(text_to_score_job)
            X = hstack([cv_vectors, job_desc_vectors])

            del cv_vectors, job_desc_vectors, text_to_score_cv, text_to_score_job
            gc.collect()

            if X.shape[1] != _model.n_features_:
                st.error(f"Erro de Incompatibilidade: Vetor gerou {X.shape[1]} features, mas modelo espera {_model.n_features_}.")
            else:
                match_score_proba = _model.predict_proba(X)[:, 1]
                df.loc[mask_valid_text, 'match_score'] = match_score_proba

            del X
            gc.collect()

        except Exception as e:
            st.error(f"Erro ao vetorizar ou calcular scores: {e}")
    else:
        st.warning("Nenhum dado válido para cálculo de score após pré-processamento.")

    df['match_score_pct'] = df['match_score'] * 100
    st.success("Cálculo de scores concluído!")
    return df

def calculate_single_score(cv_text, job_desc_text, _model, _vectorizer):
    if _model is None or _vectorizer is None:
        return None, "Recursos de Machine Learning não carregados."

    processed_cv = tokenizer(cv_text)
    processed_job_desc = tokenizer(job_desc_text)

    if not processed_cv or not processed_job_desc:
        return None, "Texto insuficiente para calcular o score. Por favor, insira textos mais descritivos."

    try:
        cv_vector = _vectorizer.transform([processed_cv])
        job_desc_vector = _vectorizer.transform([processed_job_desc])
        X = hstack([cv_vector, job_desc_vector])
    except Exception as e:
        return None, f"Erro durante a vetorização do texto: {e}"

    if X.shape[1] != _model.n_features_:
        return None, f"Erro de Incompatibilidade: Vetor gerou {X.shape[1]} features, mas modelo espera {_model.n_features_}."
        
    match_score_proba = _model.predict_proba(X)[:, 1]
    return match_score_proba[0] * 100, None

# --- FIM DAS FUNÇÕES FALTANTES ---


# --- Bloco Principal da Aplicação ---
# Garante que os recursos NLTK sejam baixados antes de qualquer outra coisa
download_nltk_resources()

# Caminhos dos Arquivos
MODEL_FILEPATH = 'lightgbm_model.pkl'
VECTORIZER_FILEPATH = 'tfidf_vectorizer.pkl'
PROCESSED_DATA_FILEPATH = 'dados_processados.parquet'

# Carrega os dados já processados
@st.cache_data
def load_preprocessed_data(path):
    try:
        df = pd.read_parquet(path)
        return df
    except FileNotFoundError:
        st.error(f"Erro: Arquivo de dados pré-processados '{path}' não encontrado.")
        return pd.DataFrame()

# Carrega os recursos e dados
model, vectorizer = load_ml_resources()
df_base_processed = load_preprocessed_data(PROCESSED_DATA_FILEPATH)

if model is not None and vectorizer is not None and not df_base_processed.empty:
    df_with_scores = calculate_match_scores_for_df(df_base_processed.copy(), model, vectorizer)
else:
    df_with_scores = df_base_processed

# --- Interface do Usuário ---
st.title("Ferramenta de Match Candidato-Vaga")
st.write("Calcule o Score de Match para novos dados ou visualize candidatos para vagas existentes.")

feature_selection = st.radio("Escolha uma funcionalidade:",
                             ('Calcular Score para Novos Dados', 'Visualizar Top Candidatos por Vaga Existente'))

if feature_selection == 'Calcular Score para Novos Dados':
    st.header("Calcular Score de Match para Novo Candidato/Vaga")
    st.write("Insira o texto do currículo e da descrição da vaga.")

    if model and vectorizer:
        new_cv_text = st.text_area("Texto do Currículo do Novo Candidato:", height=200, key='new_cv_text_area')
        new_job_desc_text = st.text_area("Texto da Descrição da Nova Vaga:", height=200, key='new_job_desc_text_area')

        if st.button("Calcular Score", key='calculate_new_score_button'):
            score, error_message = calculate_single_score(new_cv_text, new_job_desc_text, model, vectorizer)

            if error_message:
                st.error(f"Erro ao calcular score: {error_message}")
            else:
                st.subheader("Resultado do Cálculo:")
                st.metric(label="Score de Match", value=f"{score:.2f}%")
    else:
        st.warning("Recursos de ML não carregados. Não é possível calcular novos scores.")

elif feature_selection == 'Visualizar Top Candidatos por Vaga Existente':
    st.header("Visualizar Top Candidatos por Vaga Existente")

    if not df_with_scores.empty and 'id_vaga' in df_with_scores.columns:
        lista_vagas_ids = sorted(df_with_scores['id_vaga'].dropna().unique().tolist())
        
        if lista_vagas_ids:
            lista_vagas_ids.insert(0, 'Selecione uma Vaga')
            selected_job_id = st.selectbox("Escolha o ID da Vaga para ver os Top Candidatos:", lista_vagas_ids)

            if selected_job_id != 'Selecione uma Vaga':
                st.subheader(f"Candidatos Rankeados para a Vaga ID: {selected_job_id}")
                df_job_candidates = df_with_scores[df_with_scores['id_vaga'] == selected_job_id].copy()

                if not df_job_candidates.empty:
                    if 'match_score' in df_job_candidates.columns:
                        df_job_candidates_sorted = df_job_candidates.sort_values(by='match_score', ascending=False, na_position='last')
                        
                        cols_to_display = ['id_candidato', 'match_score_pct', 'situacao_candidado']
                        cols_to_display_existing = [col for col in cols_to_display if col in df_job_candidates_sorted.columns]
                        
                        if cols_to_display_existing:
                            df_display = df_job_candidates_sorted[cols_to_display_existing]

                            if 'match_score_pct' in df_display.columns:
                                df_display['match_score_pct'] = df_display['match_score_pct'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                            
                            st.dataframe(df_display, use_container_width=True)
                        else:
                            st.warning("Nenhuma coluna de resultado para exibir.")
                    else:
                        st.warning("Coluna 'match_score' não encontrada. O score não foi calculado.")
                else:
                    st.warning(f"Nenhum candidato encontrado para a Vaga ID: {selected_job_id}")
            else:
                st.info("Por favor, selecione uma vaga no menu acima.")
        else:
            st.warning("Nenhuma vaga encontrada nos dados processados.")
    else:
        st.error("Não foi possível carregar ou processar os dados. Verifique o arquivo de dados.")

# --- Barra Lateral ---
