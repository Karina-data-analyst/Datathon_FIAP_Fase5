import pandas as pd
import numpy as np
import json
import re
import string
import unicodedata
import nltk
import gc



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



def main():

    print("Iniciando pré-processamento offline...")

    try:
       
        print("Baixando recursos NLTK...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        stop_words_pt = set(nltk.corpus.stopwords.words('portuguese'))
        stop_words_eng = set(nltk.corpus.stopwords.words('english'))
        print("Recursos NLTK prontos.")

        
        print("Carregando arquivos JSON...")
        with open('vagas.json', 'r', encoding='utf-8') as f: data_vagas = json.load(f)
        vagas = [{'id': id_vaga, **conteudo.get('informacoes_basicas', {}), **conteudo.get('perfil_vaga', {}), **conteudo.get('beneficios', {})} for id_vaga, conteudo in data_vagas.items()]
        df_vagas = pd.DataFrame(vagas)

        with open('prospects.json', 'r', encoding='utf-8') as f: data_prospects = json.load(f)
        linhas_prospects = [{'id_vaga': id_vaga, 'titulo': c.get('titulo', ''), 'modalidade': c.get('modalidade', ''), **p} for id_vaga, c in data_prospects.items() for p in c.get('prospects', [])]
        df_prospects = pd.DataFrame(linhas_prospects)

        with open('applicants.json', 'r', encoding='utf-8') as f: data_applicants = json.load(f)
        candidatos = [{'id': id_c, **i.get('infos_basicas', {}), **i.get('informacoes_pessoais', {}), **i.get('informacoes_profissionais', {}), **i.get('formacao_e_idiomas', {}), 'cv_pt': i.get('cv_pt', '')} for id_c, i in data_applicants.items()]
        df_applicants = pd.DataFrame(candidatos)
        print("Arquivos JSON carregados.")

        
        print("Processando e juntando DataFrames...")
        df_applicants = df_applicants.replace('', np.nan)
        if 'outro_idioma' in df_applicants.columns:
            df_applicants['outro_idioma'] = df_applicants['outro_idioma'].replace('-', np.nan)
        df_prospects = df_prospects.replace('', np.nan)

        features_vagas = ['id', 'titulo_vaga', 'analista_responsavel', 'competencia_tecnicas_e_comportamentais']
        features_prospects = ['id_vaga', 'codigo', 'situacao_candidado']
        features_candidato = ['id', 'nome', 'cv_pt', 'telefone_celular']

        cols_vagas = [col for col in features_vagas if col in df_vagas.columns]
        cols_prospects = [col for col in features_prospects if col in df_prospects.columns]
        cols_candidato = [col for col in features_candidato if col in df_applicants.columns]

        df_prospects_features = df_prospects[cols_prospects].copy()
        df_vagas_features = df_vagas[cols_vagas].copy()
        df_applicants_features = df_applicants[cols_candidato].copy()

        if 'codigo' in df_prospects_features.columns: df_prospects_features.rename(columns={'codigo': 'id_candidato'}, inplace=True)
        if 'id' in df_vagas_features.columns: df_vagas_features.rename(columns={'id': 'id_vaga'}, inplace=True)
        if 'id' in df_applicants_features.columns: df_applicants_features.rename(columns={'id': 'id_candidato'}, inplace=True)
        if 'nome_completo' in df_applicants_features.columns: df_applicants_features.rename(columns={'nome_completo': 'nome'}, inplace=True)

        df = df_prospects_features.merge(df_vagas_features, on='id_vaga', how='left')
        df = df.merge(df_applicants_features, on='id_candidato', how='left')

        print("Filtrando vagas com apenas um candidato...")
        # Calcula quantos candidatos existem para cada vaga
        contagem_candidatos_por_vaga = df.groupby('id_vaga')['id_candidato'].transform('count')
        # Mantém apenas as linhas onde a contagem de candidatos para aquela vaga é maior que 1
        df = df[contagem_candidatos_por_vaga > 3].copy()
        print(f"DataFrame reduzido para {len(df)} linhas após a filtragem.")

        # <<< PONTO DE VERIFICAÇÃO 1 >>>
        print("\n--- PONTO 1: COLUNAS APÓS O MERGE ---")
        print(df.columns.tolist())
        print("-" * 40)

        if 'situacao_candidado' in df.columns:
            df['contratado'] = np.where(df['situacao_candidado'] == 'Contratado pela Decision', 1, 0)
        else:
            df['contratado'] = 0
        
        
        print("Aplicando tokenizer nas colunas de texto...")
        df['competencia_tecnicas_e_comportamentais_tratadas'] = df['competencia_tecnicas_e_comportamentais'].apply(lambda x: tokenizer(x, stop_words_pt, stop_words_eng))
        df['cv_tratados'] = df['cv_pt'].apply(lambda x: tokenizer(x, stop_words_pt, stop_words_eng))

        status_relevantes = [
            'Encaminhado ao Requisitante',
            'Prospect',
            'Em Entrevista Com Cliente',
            'Aprovado',
            'Contratado pela Decision'
        ]
        # Filtra o DataFrame para manter apenas as linhas com os status da lista
        df = df[df['situacao_candidado'].isin(status_relevantes)].copy()
        print(f"DataFrame reduzido para {len(df)} linhas após filtro de status.")
        
        # <<< PONTO DE VERIFICAÇÃO 2 >>>
        print("\n--- PONTO 2: COLUNAS APÓS O TOKENIZER ---")
        print(df.columns.tolist())
        print("-" * 40)

        
        print("Selecionando colunas finais...")
        colunas_necessarias = [
            'id_vaga',
            'titulo_vaga' ,
            'id_candidato', 
            'nome',
            'telefone_celular',
            'situacao_candidado', 
            'contratado', 
            'analista_responsavel',
            'competencia_tecnicas_e_comportamentais_tratadas', 
            'cv_tratados'
        ]
        colunas_finais_existentes = [col for col in colunas_necessarias if col in df.columns]
        df_final = df[colunas_finais_existentes].copy()
        
        # <<< PONTO DE VERIFICAÇÃO 3 >>>
        print("Otimizando tipos de dados do DataFrame final...")
        for col in df_final.select_dtypes(include=['int64']).columns:
            df_final[col] = pd.to_numeric(df_final[col], downcast='integer')
        for col in df_final.select_dtypes(include=['float64']).columns:
            df_final[col] = pd.to_numeric(df_final[col], downcast='float')
        print("Tipos de dados otimizados.")
        
        print("Salvando DataFrame processado em 'dados_processados.parquet'...")
        df_final.to_parquet('dados_processados.parquet', index=False)
        
        print("\nPré-processamento concluído com sucesso!")

    except Exception as e:
        print(f"ERRO: Ocorreu um erro inesperado: {e}")

if __name__ == '__main__':
    main()