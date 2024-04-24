from flask import Flask, request
from RAG import RAG

app = Flask(__name__)
RAG = RAG()

# This should receive the query to be answered
@app.route('/query', methods=['POST'])   
def get_query():
    data = request.get_json()
    
    query = data.get('query')

    response = RAG.query(query)
    answer = response["response"]
    sources = response["source"]

    return {
        'answer': answer, 
        'sources': sources
        }
# TODO: adicionar uma rota que permita ir buscar texto a um link providenciado pelo user
# TODO: adicionar a verificação da presença de hiperligações na query, se existirem primeiro vamos buscar o texto que lá está para adicionarmos ao contexto
# TODO: testar esta rota
@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    query = data.get('query')
    answer = RAG.evaluate(query)
    return {'answer': answer}

@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    query = data.get('query')
    ourAnswer, llmAnswer  = RAG.compareSolution(query)

    return {'Resposta do nosso modelo': ourAnswer, 'Resposta do LLM': llmAnswer}

@app.route('/insert', methods=['POST'])
def insert():
    data = request.get_json()
    data = data.get('data')
    RAG.insertData(data)
    return {'status': 'ok'}

if __name__ == "__main__":
    app.run(port=8000, debug=False)