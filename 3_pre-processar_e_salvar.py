import pandas as pd
import numpy as np
import json
import re
import string
import unicodedata
import nltk
import gc

# --- Funções de Pré-processamento (Consistentes com o app.py) ---

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

def tokenizer(text, stop_words_pt, stop_words_eng):
    if not isinstance(text, str):
        return ""
    text = normalize_str(text)
    text = "".join([w for w in text if not w.isdigit()])
    words = nltk.tokenize.word_tokenize(text, language='portuguese')
    words = [x for x in words if x not in stop_words_pt and x not in stop_words_eng]
    words = [y for y in words if len(y) > 2]
    return " ".join(words)

# --- Função Principal do Script ---

def main():
    """
    Script para carregar os dados JSON, processá-los e salvar o resultado
    em um arquivo Parquet otimizado para a aplicação Streamlit.
    """
    print("Iniciando pré-processamento offline...")

    try:
        # Etapa 1: Baixar recursos NLTK e carregar stopwords
        print("Baixando recursos NLTK...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        stop_words_pt = set(nltk.corpus.stopwords.words('portuguese'))
        stop_words_eng = set(nltk.corpus.stopwords.words('english'))
        print("Recursos NLTK prontos.")

        # Etapa 2: Carregar os arquivos JSON
        print("Carregando arquivos JSON...")
        with open('vagas.json', 'r', encoding='utf-8') as f:
            data_vagas = json.load(f)
        vagas = []
        for id_vaga, conteudo in data_vagas.items():
            linha = {'id': id_vaga}
            linha.update(conteudo.get('informacoes_basicas', {}))
            linha.update(conteudo.get('perfil_vaga', {}))
            linha.update(conteudo.get('beneficios', {}))
            vagas.append(linha)
        df_vagas = pd.DataFrame(vagas)

        with open('prospects.json', 'r', encoding='utf-8') as f:
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

        with open('applicants.json', 'r', encoding='utf-8') as f:
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
        print("Arquivos JSON carregados.")

        # Etapa 3: Processar e juntar os DataFrames
        print("Processando e juntando DataFrames...")
        df_applicants = df_applicants.replace('', np.nan)
        if 'outro_idioma' in df_applicants.columns:
            df_applicants['outro_idioma'] = df_applicants['outro_idioma'].replace('-', np.nan)
        df_prospects = df_prospects.replace('', np.nan)

        features_vagas = ['id', 'competencia_tecnicas_e_comportamentais']
        features_prospects = ['id_vaga', 'codigo', 'situacao_candidado']
        features_candidato = ['id', 'cv_pt']

        cols_vagas = [col for col in features_vagas if col in df_vagas.columns]
        cols_prospects = [col for col in features_prospects if col in df_prospects.columns]
        cols_candidato = [col for col in features_candidato if col in df_applicants.columns]

        df_prospects_features = df_prospects[cols_prospects].copy()
        df_vagas_features = df_vagas[cols_vagas].copy()
        df_applicants_features = df_applicants[cols_candidato].copy()

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
        
        # Etapa 4: Aplicar o tokenizer
        print("Aplicando tokenizer nas colunas de texto...")
        df['competencia_tecnicas_e_comportamentais_tratadas'] = df['competencia_tecnicas_e_comportamentais'].apply(
            lambda x: tokenizer(x, stop_words_pt, stop_words_eng)
        )
        df['cv_tratados'] = df['cv_pt'].apply(
            lambda x: tokenizer(x, stop_words_pt, stop_words_eng)
        )
        
        # Etapa 5: Selecionar colunas finais e salvar
        print("Selecionando colunas finais...")
        colunas_necessarias = [
            'id_vaga', 
            'id_candidato', 
            'situacao_candidado', 
            'contratado', 
            'competencia_tecnicas_e_comportamentais_tratadas', 
            'cv_tratados'
        ]
        # Garante que apenas colunas existentes sejam selecionadas
        colunas_finais_existentes = [col for col in colunas_necessarias if col in df.columns]
        df_final = df[colunas_finais_existentes].copy()
        
        print("Salvando DataFrame processado em 'dados_processados.parquet'...")
        df_final.to_parquet('dados_processados.parquet', index=False)
        
        print("\nPré-processamento concluído com sucesso!")

    except FileNotFoundError as e:
        print(f"ERRO: Arquivo JSON não encontrado: {e}.")
    except Exception as e:
        print(f"ERRO: Ocorreu um erro inesperado: {e}")

# --- Ponto de Entrada do Script ---
if __name__ == '__main__':
    main()