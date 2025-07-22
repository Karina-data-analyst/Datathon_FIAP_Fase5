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



@st.cache_data
def get_stopwords():
    """Carrega e retorna as listas de stopwords em português e inglês."""
    return set(stopwords.words('portuguese')), set(stopwords.words('english'))

@st.cache_resource
def download_nltk_resources():
    
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        st.error(f"Falha ao baixar recursos do NLTK: {e}")

def tokenizer(text):
    if not isinstance(text, str):
        return ""
    stop_words_pt, stop_words_eng = get_stopwords()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r" +", " ", text)
    text = "".join([w for w in text if not w.isdigit()])
    words = word_tokenize(text, language='portuguese')
    words = [x for x in words if x not in stop_words_pt and x not in stop_words_eng and len(x) > 2]
    return " ".join(words)



@st.cache_resource
def load_ml_resources():
    
    try:
        model = joblib.load(MODEL_FILEPATH)
        vectorizer = joblib.load(VECTORIZER_FILEPATH)
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"Erro: Arquivos do modelo ('{MODEL_FILEPATH}') ou vetorizador ('{VECTORIZER_FILEPATH}') não encontrados.")
        return None, None

@st.cache_data
def calculate_match_scores_for_df(df, _model, _vectorizer):
    
    if df.empty or _model is None or _vectorizer is None:
        st.warning("Não foi possível calcular scores: Dados ou recursos de ML não carregados.")
        df['match_score'] = np.nan
        df['match_score_pct'] = np.nan
        return df

    mask_valid_text = (df['cv_tratados'].str.strip().astype(bool)) & (df['competencia_tecnicas_e_comportamentais_tratadas'].str.strip().astype(bool))
    df['match_score'] = np.nan

    if mask_valid_text.any():
        try:
            cv_vectors = _vectorizer.transform(df.loc[mask_valid_text, 'cv_tratados'])
            job_desc_vectors = _vectorizer.transform(df.loc[mask_valid_text, 'competencia_tecnicas_e_comportamentais_tratadas'])
            X = hstack([cv_vectors, job_desc_vectors])

            if X.shape[1] == _model.n_features_:
                match_score_proba = _model.predict_proba(X)[:, 1]
                df.loc[mask_valid_text, 'match_score'] = match_score_proba
            else:
                st.error(f"Incompatibilidade de Features: Modelo espera {_model.n_features_}, mas os dados geraram {X.shape[1]}.")
        except Exception as e:
            st.error(f"Erro ao calcular scores: {e}")

    df['match_score_pct'] = df['match_score'] * 100
    return df

def calculate_single_score(cv_text, job_desc_text, _model, _vectorizer):
   
    if not all([_model, _vectorizer]): return None, "Recursos de ML não carregados."
    processed_cv = tokenizer(cv_text)
    processed_job_desc = tokenizer(job_desc_text)
    if not all([processed_cv, processed_job_desc]): return None, "Texto insuficiente para calcular o score."
    try:
        X = hstack([_vectorizer.transform([processed_cv]), _vectorizer.transform([processed_job_desc])])
    except Exception as e:
        return None, f"Erro na vetorização: {e}"
    if X.shape[1] != _model.n_features_: return None, f"Incompatibilidade de features: Vetor tem {X.shape[1]}, modelo espera {_model.n_features_}."
    return _model.predict_proba(X)[:, 1][0] * 100, None

# --- Bloco Principal da Aplicação ---


st.set_page_config(page_title="Decision Match AI", page_icon="🤖", layout="wide")


MODEL_FILEPATH = 'lightgbm_model.pkl'
VECTORIZER_FILEPATH = 'tfidf_vectorizer.pkl'
PROCESSED_DATA_FILEPATH = 'dados_processados.parquet'


download_nltk_resources()


@st.cache_data
def load_cached_data(path):
    
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        st.error(f"Erro: Arquivo de dados pré-processados '{path}' não encontrado. Execute o script '3_pre-processar_e_salvar.py' primeiro.")
        return pd.DataFrame()


with st.spinner("Carregando recursos e calculando scores..."):
    model, vectorizer = load_ml_resources()
    df_base_processed = load_cached_data(PROCESSED_DATA_FILEPATH)

    if model is not None and vectorizer is not None and not df_base_processed.empty:
        df_with_scores = calculate_match_scores_for_df(df_base_processed.copy(), model, vectorizer)
        
    else:
        df_with_scores = df_base_processed

# --- Interface do Usuário ---
st.title("🤖 Decision Match AI")
st.write("Uma ferramenta de IA para otimizar o processo de recrutamento, calculando o 'match' entre candidatos e vagas.")

tab1, tab2 = st.tabs(["🏆 Visualizar Top Candidatos", "🔍 Analisar Novo Match"])

def formatar_scores_com_medalhas(series_pontuacao):

    
    series_formatada = series_pontuacao.copy()
    
    
    scores_validos = series_pontuacao.dropna().sort_values(ascending=False)
    
    
    top_scores = scores_validos.unique()[:3]
    
    
    medalhas = ["🥇", "🥈", "🥉"]
    
    
    for i, score_valor in enumerate(top_scores):
        
        indices = series_pontuacao[series_pontuacao == score_valor].index
        
        series_formatada.loc[indices] = f"{medalhas[i]} {score_valor:.2f}%"
        
    
    outros_indices = series_pontuacao.index.difference(series_formatada[series_formatada.str.contains("🥇|🥈|🥉", na=False)].index)
    series_formatada.loc[outros_indices] = series_pontuacao.loc[outros_indices].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

    return series_formatada

with tab1:
    st.header("Análise de Candidatos por Vaga Existente")
    if not df_with_scores.empty and 'id_vaga' in df_with_scores.columns:
        lista_vagas_ids = sorted(df_with_scores['id_vaga'].dropna().unique().tolist())
        
        if lista_vagas_ids:
            lista_vagas_ids.insert(0, 'Selecione uma Vaga')
            selected_job_id = st.selectbox("Escolha o ID da Vaga para ver os candidatos:", lista_vagas_ids, key="select_vaga")

            if selected_job_id != 'Selecione uma Vaga':
                df_job_candidates = df_with_scores[df_with_scores['id_vaga'] == selected_job_id].copy()

                if not df_job_candidates.empty:
                    vaga_info = df_job_candidates.iloc[0]
                    titulo_vaga = vaga_info.get('titulo_vaga', 'Não informado')
                    analista_resp = vaga_info.get('analista_responsavel', 'Não informado')

                    st.write("") 
                    with st.container(border=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Nome da Vaga:**")
                            st.info(titulo_vaga)
                        with col2:
                            st.markdown("**Recrutadora Responsável:**")
                            st.info(analista_resp)
                    st.divider()

                st.subheader(f"Candidatos Rankeados para a Vaga")
                
                if not df_job_candidates.empty:
                    if 'match_score' in df_job_candidates.columns:
                        df_job_candidates_sorted = df_job_candidates.sort_values(by='match_score', ascending=False, na_position='last')
                        
                        cols_to_display = ['nome', 'telefone_celular', 'match_score_pct', 'situacao_candidado']
                        cols_to_display_existing = [col for col in cols_to_display if col in df_job_candidates_sorted.columns]
                        
                        if cols_to_display_existing:
                            df_display = df_job_candidates_sorted[cols_to_display_existing]

                           
                            if 'match_score_pct' in df_display.columns:
                                
                                df_display['match_score_pct'] = formatar_scores_com_medalhas(df_display['match_score_pct'])
                            
                            df_display.rename(columns={
                                'nome': 'Nome do Candidato',
                                'telefone_celular': 'Contato do Candidato',
                                'match_score_pct': 'Pontuação Alcançada',
                                'situacao_candidado': 'Situação Atual'
                            }, inplace=True)
                            
                            st.dataframe(df_display, use_container_width=True, hide_index=True)
                        else:
                            st.warning("Nenhuma coluna de resultado para exibir.")
                    else:
                        st.warning("A coluna 'match_score' não foi calculada.")
                else:
                    st.warning(f"Nenhum candidato para a Vaga ID: {selected_job_id}")
    else:
        st.error("Não foi possível carregar os dados. Verifique o arquivo 'dados_processados.parquet'.")

with tab2:
    st.header("Analisar Match para uma Nova Vaga")
    if model and vectorizer:
        col1, col2 = st.columns(2)
        with col1:
            new_cv_text = st.text_area("Texto do Currículo (CV):", height=250, key='new_cv_text_area')
        with col2:
            new_job_desc_text = st.text_area("Descrição da Vaga:", height=250, key='new_job_desc_text_area')
        
        if st.button("Calcular Score", key='calculate_new_score_button'):
            with st.spinner("Analisando..."):
                score, error_message = calculate_single_score(new_cv_text, new_job_desc_text, model, vectorizer)

            if error_message:
                st.error(f"Erro: {error_message}")
            elif score is not None:
                st.subheader("Resultado da Análise:")
                
                
                st.metric(label="Pontuação alcançada", value=f"{score:.2f}%")

                
                if score >= 50:
                    st.success("✅ **Forte alinhamento com a vaga.** Candidato com alto potencial, recomendado para as próximas etapas.")
                elif score >= 30:
                    st.warning("⚠️ **Alinhamento moderado.** Candidato atende a requisitos importantes. Recomenda-se análise detalhada do currículo.")
                else:
                    st.error("❌ **Baixo alinhamento com a vaga.** O perfil do candidato parece divergir dos requisitos principais.")
              
                
    else:
        st.warning("Recursos de ML não carregados.")

# --- Barra Lateral ---
st.sidebar.title("Sobre o Projeto")
st.sidebar.info(
    "Este MVP foi desenvolvido para o Datathon FIAP, aplicando técnicas de NLP e Machine Learning "
    "para criar um sistema de recomendação de candidatos."
)


st.sidebar.subheader("Desenvolvido por:")
st.sidebar.markdown("""
- **Bruno Santana Sereicikas de Azevedo** – RM358739
- **Karina Marcia da Silva** – RM359467
- **Renata Paes da silva** – RM359515
""")


st.sidebar.write("---")
st.sidebar.write("Desenvolvido para o Datathon FIAP")