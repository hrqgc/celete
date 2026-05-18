from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
import os

app = Flask(__name__)
CORS(app)

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

        print("3. Criando banco de memória LEVE (BM25)...")
        banco_de_leis = BM25Retriever.from_documents(split_docs)
        banco_de_leis.k = 4

        print("4. Conectando ao cérebro do Groq (Llama 3.1)...")
        cerebro_ia = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

        print("🚀 IA PRONTA!")
        return True
        
    except Exception as e:
        print(f"❌ Erro no setup: {e}")
        return False

@app.route('/perguntar', methods=['POST'])
def perguntar():
    global banco_de_leis, cerebro_ia
    
    # TRUQUE MÁGICO: Se a IA ainda não existir, ele liga ela agora!
    if banco_de_leis is None or cerebro_ia is None:
        sucesso = setup_ia()
        if not sucesso:
            return jsonify({"resposta": "Erro interno: Não foi possível carregar a Lei."}), 500

    data = request.json
    relato = data.get('relato', '')
    
    try:
        resultados_busca = banco_de_leis.invoke(relato)
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
