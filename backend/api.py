from flask import Flask, jsonify, request
from RAG import RAG

app = Flask(__name__)
RAG = RAG()

# This should receive the query to be answered
@app.route('/query', methods=['POST'])   
def get_query():
    data = request.get_json()
    print("Data", data)
    query = data.get('query')
    print(query)
    response = RAG.query(query)
    response = f"{response}"

    return {'response': response}

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
    app.run(port=8000)