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
    # A linha abaixo precisa estar indentada para pertencer à função
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
        # Usa quiet=True para não poluir os logs com mensagens de download
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        st.error(f"Falha ao baixar recursos do NLTK: {e}")

# <<< MELHORIA 1: Otimização do Tokenizer >>>
# As stopwords agora são carregadas uma única vez e guardadas em cache.
stop_words_pt, stop_words_eng = get_stopwords()

def tokenizer(text):
    if not isinstance(text, str):
        return ""
    
    # Carrega as stopwords aqui. Graças ao cache, isso só acontece de verdade uma vez.
    stop_words_pt, stop_words_eng = get_stopwords()
    
    text = normalize_str(text)
    text = "".join([w for w in text if not w.isdigit()])
    
    words = word_tokenize(text, language='portuguese')
    words = [x for x in words if x not in stop_words_pt and x not in stop_words_eng]
    words = [y for y in words if len(y) > 2]
    # A linha abaixo precisa estar indentada para pertencer à função
    return " ".join(words)
# --- Funções de Carregamento e Processamento de Dados ---
@st.cache_data
def load_and_process_data(vagas_path, prospects_path, applicants_path):
    try:
        # Carregamento dos JSONs
        with open(vagas_path, 'r', encoding='utf-8') as f:
            data_vagas = json.load(f)
        vagas = []
        for id_vaga, conteudo in data_vagas.items():
            linha = {'id': id_vaga}
            linha.update(conteudo.get('informacoes_basicas', {}))
            linha.update(conteudo.get('perfil_vaga', {}))
            linha.update(conteudo.get('beneficios', {}))
            vagas.append(linha)
        df_vagas = pd.DataFrame(vagas)

        with open(prospects_path, 'r', encoding='utf-8') as f:
            data_prospects = json.load(f)
        linhas_prospects = []
        for id_vaga, conteudo in data_prospects.items():
            titulo = conteudo.get('titulo', '')
            modalidade = conteudo.get('modalidade', '')
            prospects = conteudo.get('prospects', [])
            for prospect in prospects:
                linha = {'id_vaga': id_vaga, 'titulo': titulo, 'modalidade': modalidade}
                linha.update(prospect)
                linhas_prospects.append(linha)
        df_prospects = pd.DataFrame(linhas_prospects)

        with open(applicants_path, 'r', encoding='utf-8') as f:
            data_applicants = json.load(f)
        candidatos = []
        for id_candidato, info in data_applicants.items():
            linha = {'id': id_candidato}
            linha.update(info.get('infos_basicas', {}))
            linha.update(info.get('informacoes_pessoais', {}))
            linha.update(info.get('informacoes_profissionais', {}))
            linha.update(info.get('formacao_e_idiomas', {}))
            linha['cv_pt'] = info.get('cv_pt', '')
            candidatos.append(linha)
        df_applicants = pd.DataFrame(candidatos)

        # <<< MELHORIA 2: Correção dos Avisos (FutureWarning) do Pandas >>>
        df_applicants = df_applicants.replace('', np.nan)
        if 'outro_idioma' in df_applicants.columns:
            df_applicants['outro_idioma'] = df_applicants['outro_idioma'].replace('-', np.nan)
        df_prospects = df_prospects.replace('', np.nan)

        # Seleção de colunas
        features_vagas = ['id', 'competencia_tecnicas_e_comportamentais']
        features_prospects = ['id_vaga', 'codigo', 'situacao_candidado']
        features_candidato = ['id', 'cv_pt']

        cols_vagas = [col for col in features_vagas if col in df_vagas.columns]
        cols_prospects = [col for col in features_prospects if col in df_prospects.columns]
        cols_candidato = [col for col in features_candidato if col in df_applicants.columns]

        df_prospects_features = df_prospects[cols_prospects].copy()
        df_vagas_features = df_vagas[cols_vagas].copy()
        df_applicants_features = df_applicants[cols_candidato].copy()

        # Renomeação e merges
        if 'codigo' in df_prospects_features.columns:
            df_prospects_features.rename(columns={'codigo': 'id_candidato'}, inplace=True)
        if 'id' in df_vagas_features.columns:
            df_vagas_features.rename(columns={'id': 'id_vaga'}, inplace=True)
        if 'id' in df_applicants_features.columns:
            df_applicants_features.rename(columns={'id': 'id_candidato'}, inplace=True)

        df = df_prospects_features.merge(df_vagas_features, on='id_vaga', how='left')
        df = df.merge(df_applicants_features, on='id_candidato', how='left')
        
        if 'situacao_candidado' in df.columns:
            df['contratado'] = np.where(df['situacao_candidado'] == 'Contratado pela Decision', 1, 0)
        else:
            df['contratado'] = 0

        # Aplicação do tokenizer
        df['competencia_tecnicas_e_comportamentais_tratadas'] = df['competencia_tecnicas_e_comportamentais'].apply(tokenizer)
        df['cv_tratados'] = df['cv_pt'].apply(tokenizer)
        
        # Otimização de memória
        del df_vagas, df_prospects, df_applicants
        del df_prospects_features, df_vagas_features, df_applicants_features
        gc.collect()
        
        return df

    except FileNotFoundError as e:
        st.error(f"Erro: Arquivo JSON não encontrado: {e}.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar ou processar dados JSON: {e}")
        st.exception(e)
        return pd.DataFrame()

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

    st.info("Calculando Scores de Match para os dados carregados...")
    df_temp = df.copy()

    df_to_score = df_temp[
        (df_temp['cv_tratados'].str.strip().astype(bool)) &
        (df_temp['competencia_tecnicas_e_comportamentais_tratadas'].str.strip().astype(bool))
    ].copy()

    if df_to_score.empty:
        st.warning("Nenhum dado válido para cálculo de score após pré-processamento.")
        df_temp['match_score'] = np.nan
        df_temp['match_score_pct'] = np.nan
        return df_temp

    try:
        cv_vectors = _vectorizer.transform(df_to_score['cv_tratados'])
        job_desc_vectors = _vectorizer.transform(df_to_score['competencia_tecnicas_e_comportamentais_tratadas'])
        X = hstack([cv_vectors, job_desc_vectors])
    except Exception as e:
        st.error(f"Erro ao vetorizar dados: {e}")
        df_temp['match_score'] = np.nan
        df_temp['match_score_pct'] = np.nan
        return df_temp

    if X.shape[1] != _model.n_features_:
        st.error(f"Erro de Incompatibilidade: Vetor gerou {X.shape[1]} features, mas modelo espera {_model.n_features_}.")
        df_temp['match_score'] = np.nan
        df_temp['match_score_pct'] = np.nan
        return df_temp

    match_score_proba = _model.predict_proba(X)[:, 1]
    df_to_score['match_score'] = match_score_proba

    scores_calculados = df_to_score[['id_vaga', 'id_candidato', 'match_score']]

    df_final = df_temp.merge(scores_calculados, on=['id_vaga', 'id_candidato'], how='left', suffixes=('_antigo', ''))
    
    if 'match_score_antigo' in df_final.columns:
        df_final.drop(columns=['match_score_antigo'], inplace=True)
    
    df_final['match_score_pct'] = df_final['match_score'] * 100

    st.success("Scores de Match calculados com sucesso!")
    return df_final

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

# --- Bloco Principal da Aplicação ---
download_nltk_resources()

VAGAS_FILEPATH = 'vagas.json'
PROSPECTS_FILEPATH = 'prospects.json'
APPLICANTS_FILEPATH = 'applicants.json'
MODEL_FILEPATH = 'lightgbm_model.pkl'
VECTORIZER_FILEPATH = 'tfidf_vectorizer.pkl'

model, vectorizer = load_ml_resources()
df_base_processed = load_and_process_data(VAGAS_FILEPATH, PROSPECTS_FILEPATH, APPLICANTS_FILEPATH)

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

    # <<< MELHORIA 3: Verificação de Colunas Mais Robusta >>>
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
        st.error("Não foi possível carregar ou processar os dados. Verifique os arquivos JSON.")

# --- Barra Lateral ---
st.sidebar.subheader("Como Rodar")
st.sidebar.write("1. Salve este código como `app.py`.")
st.sidebar.write(f"2. Certifique-se de que os arquivos de dados e modelos estão no mesmo diretório.")
st.sidebar.write("3. Abra um terminal no diretório.")
st.sidebar.write("4. Execute: `streamlit run app.py`")
st.sidebar.write("---")
st.sidebar.write("Desenvolvido para o Datathon")