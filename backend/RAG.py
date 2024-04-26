#Utils
from dotenv import load_dotenv
import os
import os.path
import json
import datetime
from datasets import load_dataset
import time

#RAG pipeline

# Alter from the llama index to the https://docs.together.ai/docs/quickstart
# Together documentation: https://github.com/togethercomputer/together-python
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain_together import Together

from deepeval.metrics import GEval

from ragas import evaluate
from ragas.embeddings import HuggingfaceEmbeddings
from ragas.metrics import(
    answer_relevancy,
    answer_correctness,
    faithfulness
)

#Database
from DBController import DBController

import numpy as np
import spacy


class TogetherModel():

    def __init__(self):
        self.model = TogetherLLM(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            api_key = os.getenv("TogetherAI_API_KEY")
        )
    
    def load_model(self):
        return self.model

    def generate(self, prompt):
        answer = self.model.complete(prompt)
        return answer
    
    # Todo verify this method
    async def a_generate(self, prompt):
        answer = await self.model.complete(prompt)
        return answer
    
    def get_model_name(self):
        return "MistraLAI Mixtral 8x7B Instruct v0.1"

#TODO Make run requirements and install spacy python3 -m spacy download pt_core_news_sm

class RAG():
    def __init__(self):
        load_dotenv()
        # TODO: Usar o ficheiro parties.json para fazer o mapeamento entre partidos e siglas e outras informações caso seja necessário
        partiesFile = open("data/parties.json")
        self.partiesFile = json.load(partiesFile)
        status = os.system("export TOKENIZERS_PARALLELISM=true")
        print("Status", status)
        # Load the models and the database controller
        self.llm = self.loadModel()
        self.embedingModel = self.loadEmbeddingModel()
        self.DBController = DBController(self.modelContextWindow)
        
        # self.testEmbedings()
        # self.testModel()

        # Ingest the data and insert into the database
        # self.dataIngestion()

        # For pipeline evaluation
        # self.evaluate()

    def partiesInQuery(self, query):
        # Provavelmente adicionar um modelo para verificar a verossimilança entre nomes de partidos

        partidosAFiltrar = []
        query = query.lower()

        for partido in self.partiesFile["partidos"]:
            siglas = self.partiesFile["partidos"][partido]["siglas"]
            for sigla in siglas:
                if sigla.lower() in query:
                    partidosAFiltrar.append(partido)
                    break

        # Caso não tenha nenhum partido na query, procurar em todos
        if not partidosAFiltrar:
            uniquePartidos = list(self.partiesFile["partidos"].keys())
            # Lets get the unique parties
            partidosAFiltrar = list(set(uniquePartidos))

        return partidosAFiltrar
        
    # TODO: Send the query size to the database controller when querying in order to get the context
    def query(self, query, evaluate=False):
        try:
            # TODO: identificar a presença de partidos e temáticas nas queries de forma a correr cada query à base de dados com base nesses partidos
            # TODO: Fazer um mapa entre partidos e as possíveis siglas, também podemos adicionar alguma história para cada partido

            # Exemplo: Quais as medidas do ps e da ad para a economia?
            # Partidos: PS, AD
            # Temáticas: Economia
            # Query na base de dados ao PS sobre a economia
            # Query na base de dados à AD sobre a economia
            # Passar tudo como contexto
            filters = None
            parties = self.partiesInQuery(query)

            results = {}

            for party in parties:
                
                filters = {"partido": {"$eq" : party}}

                embededQuery = self.generateEmbeddings(query)
                
                extraContext = self.DBController.runQuery(embededQuery, filters, query)
            
                minimumConfidence = 0.80
                contextAdd = []
                sources = []
                
                # print(extraContext)
                for confidenceLevel in extraContext:
                    if confidenceLevel > minimumConfidence:
                        for (context, source) in extraContext[confidenceLevel]:
                            contextAdd.append(context)
                            sources.append(source)

                print("Confidence level", np.average(list(extraContext.keys())))

                if len(extraContext)==0:
                    contextAdd = f"Para o partido {party} não encontrei nada sobre isso."
                # return only the unique sources 
                sources = list(set(sources))
                results[party] = ("\n".join(contextAdd), sources)


            # If the retrieved context is non existant or the confidence in the answer is too low then 
            # Dont answer
            # print("Extra context", extraContext)

            # TODO:Prompt engineering to enhance the response
            # Vamos usar one shot learning para melhorar a resposta
            # https://ritikjain51.medium.com/llms-mastering-llm-responses-through-advanced-prompt-engineering-strategies-25c029d504b2

            # Vamos usar https://www.promptingguide.ai/techniques/cot visto que elimina 

            # TODO: Fazer a query diferente para um único partido como filtro e múltiplos, o one shot learning fica diferente
            # TODO: Analisar o tamanho do contexto e ver se é necessário fazer a query de outra forma tipo sumarizar
            # Ou uma para cada partido e depois juntar tudo
            query = f"""
                        És um assistente para responder a questões sobre o plano eleitoral para 2024.
                        Dá me a resposta à seguinte questão em Português de Portugal!\n
                        Caso não tenhas a certeza das respostas diz que não sabes.\n
                        Com base única e exclusivamente no seguinte contexto do plano eleitoral para 2024.
                        Contexto: {[f"O partido {party} diz o seguinte: {contextAdd}" for party, (contextAdd, _) in results.items()]}
                        
                        Pergunta: {query}
                    """
            
            # print("Querying with query: ",query)
            # response = self.llm.complete(query)
            response = self.llm.complete(query)

            # Lets log the query and the response to a file
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            filename = f"query_{date}.json"
        
            with open(f"logs/{filename}", "w") as outfile:
                json.dump(
                    {"query": query, 
                     "response": str(response), 
                     "context": [
                         {
                            f"{party}" : f"{contextAdd}" for party, (contextAdd, _) in results.items()
                        }
                     ]}, 
                     outfile)

            if not evaluate:
                return {
                        "response" : str(response),
                        "source" : [source for (_, source) in results.values()]
                }
            
            return{
                "response": str(response),
                "source": [source for (_, source) in results.values()],
                "context" : contextAdd
            }
                
        
        except Exception as e:
            print("Error querying", e)

        
    # TODO: https://docs.llamaindex.ai/en/latest/examples/low_level/ingestion/
    def dataIngestion(self):
        try:
            print("Data Ingestion")
            #Lets load the data
            self.loadData()
        except Exception as e:
            print("Error ingesting data", e)


    def process(self, text):
                doc = self.nlp(text)
                sents = list(doc.sents)
                vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])

                return sents, vecs
    
    def cluster_text(self,sents, vecs, threshold):
        clusters = [[0]]
        for i in range(1, len(sents)):
            if np.dot(vecs[i], vecs[i-1]) < threshold:
                clusters.append([])
            clusters[-1].append(i)
        
        return clusters

    # TODO: Edit this in order to account for overlapping
    def chunkFile(self, file, tokensPerChunk, overlap):
        try:
                        
            # Load the Spacy model
            self.nlp = spacy.load('pt_core_news_lg')

            # Initialize the clusters lengths list and final texts list
            clusters_lens = []
            final_texts = []

            # Process the chunk
            threshold = 0.3
            sents, vecs =self.process(file)

            # Cluster the sentences
            clusters = self.cluster_text(sents, vecs, threshold)

            for cluster in clusters:
                cluster_txt = ' '.join([sents[i].text for i in cluster])
                cluster_len = len(cluster_txt)
                
                # Check if the cluster is too short
                if cluster_len < 60:
                    continue
                
                # Check if the cluster is too long
                elif cluster_len > tokensPerChunk:
                    threshold = 0.6
                    sents_div, vecs_div = self.process(cluster_txt)
                    reclusters = self.cluster_text(sents_div, vecs_div, threshold)
                    
                    for subcluster in reclusters:
                        div_txt = ' '.join([sents_div[i].text for i in subcluster])
                        div_len = len(div_txt)
                        
                        if div_len < 60 or div_len > tokensPerChunk:
                            continue
                        
                        clusters_lens.append(div_len)
                        final_texts.append(div_txt)
                        
                else:
                    clusters_lens.append(cluster_len)
                    final_texts.append(cluster_txt)
                
            print(f"Number of chunks: {len(final_texts)}")
            # for text in final_texts:
            #     print("Chunk:", text)
                
            return final_texts
        
        except Exception as e:
            print("Error chunking file", e)
        pass


    def loadData(self):
        try:
            print("Load Data")

            #Lets load the data
            docs = []
            i=0
            for party in os.listdir("data"):
                if not os.path.isdir(f"data/{party}"):
                    continue

                for doc in os.listdir(f"data/{party}"):
                    if ".txt" not in doc:
                        continue

                    docName = doc.split(".")[0]
                    docs.append(doc)
                    # document = loader.load(file_path=f"./data/{doc}")

                    data = open(f"./data/{party}/{doc}")
                    data = data.readlines()
                    data = "\n".join(data)

                    chunkedData = self.chunkFile(file=data, tokensPerChunk=2000, overlap=50)
                    
                    # In the chunker limit the chunking token size to the number of tokens that the embeder can process
                    cleanName = docName.replace(" ", "_").replace(":", "_").replace("?", "_").replace("!", "_").replace("(", "_").replace(")", "_").replace(",", "_").replace(".", "_")
                    print("Data chunked")
                    for idx in range(len(chunkedData)):
                        text_chunk = chunkedData[idx]
                        
                    

                        nodeEmbeding = self.generateEmbeddings(text_chunk)
                        time.sleep(0.2)
                        # print(f"Embeding size: {len(nodeEmbeding)}")

                        # TODO: Convém indexar o link do documento original para meter nos metadados
                        # TODO: Fix no id para ir buscar contexto adicional 

                        payload = {
                            "id": str(i),
                            "values": nodeEmbeding,
                            "metadata": {
                                "chunk_id": idx,
                                "document_id": f"{party}_{cleanName}",
                                "partido": party,
                                "assunto" : docName,
                                "texto" : text_chunk,
                                "source": self.partiesFile["partidos"][party]["source"]
                            }
                        }
                        
                        self.DBController.insert(payload)
                        i+=1
                
                print(f"Loaded {len(docs)} documents into {party}")
            print("Data loaded")

        except Exception as e:
            print("Error loading data", e)


    def generateEmbeddings(self, data):
        try:
            data = self.embedingModel.get_text_embedding(data)

            return data
        
        except Exception as e:
            print("Error generating embeddings", e)

    
    def evaluate(self, testset={}):
        print("Evaluating the pipeline")
        try:
            # Lets load the jsonfile with the testset
            if testset == {}:
                testset = open("tests.json")
                testset = json.load(testset)
                testset = testset["tests"]

            results = []

            criticModel = Together(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                temperature=0.1,
                together_api_key = os.getenv("TogetherAI_API_KEY"),
                max_tokens = 512,
                # Answer structure
            )

            embedingModel = HuggingfaceEmbeddings(
                model_name = "BAAI/bge-small-en"
            )


            metrics = [
                # context_precision,
                answer_relevancy,
                answer_correctness,
                faithfulness
            ]

            evaluationResults = []

            


            for test in testset:
                name = test["name"]
                question = test["question"]
                groundTruth = test["groundTruth"]

                answer = self.query(question, evaluate = True)
                ourAnswer = answer["response"]
                context = answer["context"]
                
                evaluation_template = f"""
                    Classifica a resposta: "{ourAnswer}" para a pergunta "{question}" dado apenas o contexto fornecido pelo seguinte contexto: {"".join(context)}.
                    O valor real da pergunta deverá ser "{groundTruth}".
                    A classificação deve ser entre 1 (pontuação mais baixa) e 10 (pontuação mais alta), e deve conter uma explicação máxima de uma frase da classificação.
                    A classificação deve ser baseada na relevância da resposta e o quão correta ela está considerando que a resposta foi APENAS baseada no contexto, e nada mais.
                    A resposta deve ter apenas a classificação do tipo:
                    Relevancia: x/10
                    Correção: x/10
                    """

                modelAnswer = criticModel.invoke(evaluation_template)
                

                ourAnswer = ourAnswer.replace("\n", " ").replace("Resposta:", "")
                
                result = {
                    "question": question,
                    "answer": ourAnswer,
                    "contexts": ["\n".join(context[0:5])],
                    "ground_truth": groundTruth
                }

                # Lets create a json file to store the evaluation:
                date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
                filename = f"results_{date}.json"

                with open(f"evaluation/{filename}", "w") as outfile:
                    json.dump(result, outfile)

                result = load_dataset("json", data_files=f"evaluation/{filename}")
                
                results = evaluate(
                    result["train"],
                    metrics=metrics,
                    llm = criticModel,
                    embeddings = embedingModel,
                    raise_exceptions=False
                )
                
                results = results.to_pandas()

                # Lets overwrite the json file now with the answers from both the critic llm and the RAGAS evaluation
                # Probably alter the order for better visualization in the file

                # TODO: verify if any of the results are Nan, if so replace by 0 
                answer_relevancy_result = results["answer_relevancy"].iloc[0]
                if np.isnan(answer_relevancy_result):
                    answer_relevancy_result = 0
                answer_correctness_result = results["answer_correctness"].iloc[0]
                if np.isnan(answer_correctness_result):
                    answer_correctness_result = 0
                answer_faithfulness_result = results["faithfulness"].iloc[0]
                if np.isnan(answer_faithfulness_result):
                    answer_faithfulness_result = 0

                with open(f"evaluation/{filename}", "w") as outfile:
                    payload = {
                        "test_name": name,
                        "question" : question,
                        "criticAnswer": modelAnswer,
                        "evaluation": {
                            "answer_relevancy": float(answer_relevancy_result),
                            "answer_correctness": float(answer_correctness_result),
                            "answer_faithfulness": float(answer_faithfulness_result)
                        },
                        "groundTruth": groundTruth,
                        "ourAnswer": ourAnswer,
                        "context": context
                    }
                    json.dump(payload, outfile)

                    evaluationResults.append(payload)

            print("Evaluation done")
            return {"results" : evaluationResults}

        except Exception as e:
            print("Error evaluating", e)



    def loadModel(self):
        try: 
            togetherai_api_key = os.getenv("TogetherAI_API_KEY")
            # print("TogetherAI_API_KEY", togetherai_api_key)
            # Context window 32768
            
            # modelName = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            self.modelContextWindow = 32700
            # llm = Together(
            #     model=modelName,
            #     temperature=0.5,
            #     together_api_key = togetherai_api_key,
            #     max_tokens = 1024
            #     )
            llm = TogetherLLM(
                model= "mistralai/Mixtral-8x7B-Instruct-v0.1",
                api_key = togetherai_api_key
            )
            print("Model loaded")
            return llm
        except Exception as e:
            print("Error loading model", e)
            return None

    # TODO: Load embedding model: https://huggingface.co/spaces/mteb/leaderboard
    # should load: voyage-lite-02-instruct, max tokens 4000
    # Embedding dimension: 1024
    # OCUPA 4GB
    def loadEmbeddingModel(self):
        try: 
            # togetherai_api_key = os.getenv("TogetherAI_API_KEY")

            embed_model = HuggingFaceEmbedding(
                model_name = "BAAI/bge-small-en"
            )

            print("Embedding model loaded")
            return embed_model
        except Exception as e:
            print("Error loading embedding model", e)
            return None
    
    # within size to query the model
    def compareSolution(self, query):
        try:
            ourAnswer = self.query(query)["response"]
        
            llmAnswer = self.llm.complete(query)
        
            return (ourAnswer, str(llmAnswer))

        except Exception as e:
            print("Error comparing solutions", e)
            return ("Ocorreu um erro", "Ocorreu um erro")


    # Testing
        
    def testEmbedings(self):
        try:
            print("Testing Embeddings")
            nodeEmbedding = self.embedingModel.get_text_embedding("Teste")
            print(f"Embeding size: {len(nodeEmbedding)}")
        except Exception as e:
            print("Error testing embeddings", e)

    def testModel(self):
        try:
            print("Testing Model")
            print(self.llm.complete("Hello, what is your name?"))
        except Exception as e:
            print("Error testing model", e)    


    def testQueries(self):
        option = input("What is the query you want to make?")
        #Quais as faculdades do banco de portugal?
        while(option!=""):
            try:
                print("Testing Queries")
                print(f"LLM answer: {self.llm.invoke(option)}")

                print(f"Our answer: {self.DBController.runQuery(option)}")
                option = input("What is the query you want to make?")
            except Exception as e:
                print("Error testing queries", e)
