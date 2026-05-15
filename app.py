from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os

app = Flask(__name__)
CORS(app)

# Variáveis globais para armazenar nossa memória e a IA
banco_de_leis = None
cerebro_ia = None

def setup_ia():
    global banco_de_leis, cerebro_ia
    try:
        print("1. Lendo o PDF da CLT...")
        loader = PyPDFLoader("clt.pdf")
        docs = loader.load()

        print("2. Fatiando as leis...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        print("3. Criando banco de memória local...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        banco_de_leis = FAISS.from_documents(split_docs, embeddings)

        print("4. Conectando ao cérebro do Groq (Llama 3.1)...")
        cerebro_ia = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

        print("🚀 IA PRONTA!")
        
    except Exception as e:
        print(f"❌ Erro no setup: {e}")

# FORÇAMOS A IA A LIGAR JUNTO COM O SERVIDOR NA NUVEM
setup_ia()

@app.route('/perguntar', methods=['POST'])
def perguntar():
    if not banco_de_leis or not cerebro_ia:
        return jsonify({"resposta": "Aguarde, a IA ainda está carregando..."}), 503
    
    data = request.json
    relato = data.get('relato', '')
    
    try:
        resultados_busca = banco_de_leis.similarity_search(relato, k=4)
        contexto_da_lei = "\n\n".join([doc.page_content for doc in resultados_busca])
        
        prompt_manual = f"""Você é um assistente jurídico focado na CLT brasileira.
        Analise o relato do trabalhador com base APENAS nos trechos da lei abaixo.
        
        REGRAS DE FORMATAÇÃO:
        1. Seja o mais curto, direto e resumido possível. Sem enrolação.
        2. Liste os artigos da CLT correspondentes em formato de lista (usando o símbolo • antes de cada artigo).
        3. Adicione um aviso final curto para procurar um advogado.

        TRECHOS DA LEI ENCONTRADOS:
        {contexto_da_lei}

        RELATO DO TRABALHADOR:
        {relato}

        Análise:"""
        
        resposta_ia = cerebro_ia.invoke(prompt_manual)
        return jsonify({"resposta": resposta_ia.content})
        
    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({"resposta": "Erro ao processar sua pergunta."}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=False)
