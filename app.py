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

# ===== A MÁGICA ACONTECE AQUI =====
# Agora o Python é quem entrega a sua página do site!
@app.route('/', methods=['GET'])
def home():
    html_do_site = """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Consultor CLT</title>
        <style>
            :root { --bg-color: #131314; --text-color: #e3e3e3; --input-bg: #1e1f20; --accent: #4285f4; }
            body, html { margin: 0; padding: 0; height: 100%; background-color: var(--bg-color); color: var(--text-color); font-family: 'Segoe UI', Roboto, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; overflow: hidden; }
            #chat-container { width: 100%; max-width: 800px; height: 70vh; overflow-y: auto; display: none; padding: 20px; margin-bottom: 100px; }
            .input-wrapper { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 90%; max-width: 700px; transition: all 0.6s cubic-bezier(0.25, 1, 0.5, 1); }
            .input-wrapper.active { top: 90%; }
            .search-area { background: var(--input-bg); border-radius: 30px; padding: 15px 25px; display: flex; align-items: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
            input { background: transparent; border: none; color: white; flex: 1; font-size: 18px; outline: none; }
            button { background: none; border: none; color: var(--accent); cursor: pointer; font-weight: bold; }
            .msg { margin: 20px 0; line-height: 1.6; }
            .user { color: #8ab4f8; font-weight: bold; }
            .ai { background: #28292a; padding: 15px; border-radius: 15px; }
            h1 { font-weight: 400; font-size: 2rem; margin-bottom: 20px; transition: opacity 0.3s; }
        </style>
    </head>
    <body>
        <h1 id="title">Como posso ajudar com a CLT hoje?</h1>
        <div id="chat-container"></div>
        <div class="input-wrapper" id="inputBox">
            <div class="search-area">
                <input type="text" id="entrada" placeholder="Explique sua situação trabalhista..." onkeypress="if(event.key === 'Enter') enviar()">
                <button onclick="enviar()">➤</button>
            </div>
        </div>
        <script>
            async function enviar() {
                const input = document.getElementById('entrada');
                const chat = document.getElementById('chat-container');
                const wrapper = document.getElementById('inputBox');
                const title = document.getElementById('title');
                const texto = input.value;

                if(!texto) return;

                title.style.opacity = '0';
                wrapper.classList.add('active');
                chat.style.display = 'block';
                chat.innerHTML += `<div class="msg"><span class="user">Você</span><br>${texto}</div>`;
                input.value = '';

                try {
                    // Agora o fetch aponta direto para a própria raiz do site
                    const res = await fetch('/perguntar', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ relato: texto })
                    });
                    const data = await res.json();
                    chat.innerHTML += `<div class="msg ai"><strong>Consultor CLT:</strong><br>${data.resposta}</div>`;
                    chat.scrollTop = chat.scrollHeight;
                } catch (err) {
                    chat.innerHTML += `<p>Erro ao conectar com o servidor.</p>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return html_do_site
# ==================================

@app.route('/perguntar', methods=['POST'])
def perguntar():
    global banco_de_leis, cerebro_ia
    
    # Se a IA ainda não existir, ele liga ela agora!
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
