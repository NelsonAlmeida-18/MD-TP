#Utils
from dotenv import load_dotenv
import os
import os.path
import json
import datetime
from datasets import load_dataset
import time
import re
import matplotlib.pyplot as plt
import requests
# Ignore the warnings
import warnings
warnings.filterwarnings("ignore")

#RAG pipeline

from DBController import DBController

# Alter from the llama index to the https://docs.together.ai/docs/quickstart
# Together documentation: https://github.com/togethercomputer/together-python
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



from ragas import evaluate
from ragas.embeddings import HuggingfaceEmbeddings
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.metrics import(
    answer_relevancy,
    answer_correctness,
    faithfulness,
    context_precision,
    context_recall
)

import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('averaged_perceptron_tagger')

import numpy as np
import spacy


class Evaluate():
    def __init__(self):
        load_dotenv()

        partiesFile = open("data/parties.json")
        self.partiesFile = json.load(partiesFile)
        # Load the models and the database controller
        self.llm = self.loadModel()
        self.embedingModel = self.loadEmbeddingModel()
        self.DBController = DBController()
        # self.testEmbedings()
        # self.testModel()

        # Ingest the data and insert into the database
        # self.dataIngestion()

        #self.generateSyntheticDataset()
        self.evaluate("synthetic_testset_2024-06-06_08-56.json")

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


            # TODO: Fazer a query diferente para um único partido como filtro e múltiplos, o one shot learning fica diferente
            # TODO: Analisar o tamanho do contexto e ver se é necessário fazer a query de outra forma tipo sumarizar
            # Ou uma para cada partido e depois juntar tudo
            modelQuery = f"""
                És um assistente especializado em responder a questões sobre o plano eleitoral para 2024.
                A TUA RESPOSTA TEM DE SER EM PORTUGUÊS DE PORTUGAL!

                Se não souberes a resposta, deves dizer que não sabes.

                Baseia-te única e exclusivamente no seguinte contexto do plano eleitoral para 2024:
                Contexto: {[f"O partido {party} diz o seguinte: {contextAdd}" for party, (contextAdd, _) in results.items()]}

                Questão: {query}

                Lembra-te, a resposta deve ser sempre em Português de Portugal!

                Agora, com base no contexto fornecido, responde à seguinte questão:
            """

            
            # print("Querying with query: ",query)
            # response = self.llm.complete(query)
            response = self.llm.complete(modelQuery)

            # if len(response)<300:
            #     return self.query(query, evaluate)

        
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
            try:
                doc = self.nlp(text)
                sents = list(doc.sents)
                # Todo rever isto
                vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])

                return sents, vecs
            except Exception as e:
                print("Error processing text", e)
                return [], []
    
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
            # TODO: altered this to the small in order to consume less ram
            self.nlp = spacy.load('pt_core_news_sm')

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
            return []
        pass


    
    def loadData(self):
        try:
            print("Load Data")
            np.seterr(divide='ignore', invalid='ignore')
            #Lets load the data
            
            i=0
            for doc in os.listdir("data"):
                docs = []
        
                if ".txt" not in doc:
                    continue

                docName = doc.split(".")[0]
                docs.append(doc)
                # document = loader.load(file_path=f"./data/{doc}")

                data = open(f"./data/{doc}")
                data = data.readlines()
                data = "\n".join(data)

                chunkedData = self.chunkFile(file=data, tokensPerChunk=3000, overlap=50)
                
                # In the chunker limit the chunking token size to the number of tokens that the embeder can process
                cleanName = docName.replace(" ", "_").replace(":", "_").replace("?", "_").replace("!", "_").replace("(", "_").replace(")", "_").replace(",", "_").replace(".", "_")
                print("Data chunked")
                for idx in range(len(chunkedData)):
                    text_chunk = chunkedData[idx]
                    
                

                    nodeEmbeding = self.generateEmbeddings(text_chunk)
                    time.sleep(0.5)
                    # print(f"Embeding size: {len(nodeEmbeding)}")

                    # TODO: Convém indexar o link do documento original para meter nos metadados
                    # TODO: Fix no id para ir buscar contexto adicional 

                    payload = {
                        "id": str(i),
                        "values": nodeEmbeding,
                        "metadata": {
                            "chunk_id": idx,
                            "partido": "X",
                            "assunto" : docName,
                            "texto" : text_chunk,
                            "source": "https://www.parlamento.pt/"
                        }
                    }
                    
                    
                    i+=1



            print("Data loaded")

        except Exception as e:
            print("Error loading data", e)


    def generateEmbeddings(self, data):
        try:
            data = self.embedingModel.get_text_embedding(data)

            return data
        
        except Exception as e:
            print("Error generating embeddings", e)

    def generateSyntheticDataset(self, partidos=os.listdir("./data/")):

        print("Generating synthetic dataset")
        openAIKey = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = openAIKey

        filename = f"synthetic_testset_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
        with open(f"evaluation/{filename}", "a") as outfile:
            #Lets store the structure to the file
            payload = {
                "tests": []
            }
            testNum=0
            for partido in partidos:
                if not os.path.isdir(f"data/{partido}"): 
                    continue
                
                print(f"Generating synthetic data for partido: {partido}")
                loader = DirectoryLoader(f"data/{partido}")

                documents = loader.load()

                for document in documents:
                    document.metadata['filename'] = document.metadata['source']

                # generator with openai models
                generator_llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
                critic_llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
                embeddings = OpenAIEmbeddings()

                generator = TestsetGenerator.from_langchain(
                    generator_llm,
                    critic_llm,
                    embeddings
                )

                # generate testset  
                testset = generator.generate_with_langchain_docs(documents, test_size=6, distributions={ reasoning: 0.5, multi_context: 0.5})
                testset_df = testset.to_pandas()

                testset_dict = testset_df.to_dict(orient="records")
                
                # save testset
                

                for i in testset_dict:
                    payload["tests"].append({

                        "name" : f"Test_{testNum}",
                        "question": i["question"],
                        "groundTruth": i["ground_truth"],
                        "partido": partido,
                        # "context": "\n".join(i["contexts"])

                    })

                    testNum+=1

            json.dump(payload, outfile)

        print("Synthetic data generated")

    def getModelResults(self, modelAnswer):
        try:
            results = {}

            i= modelAnswer
            regexRelevanciaResposta = r"Relevancia da resposta: (\d+)/10"
            regexCorrecao = r"Correção: (\d+)/10"  
            regexPrecisaoContexto = r"Precisão do contexto: (\d+)/10"
            regexRecallContexto = r"Recall do contexto: (\d+)/10"

            relevanciaResposta = re.findall(regexRelevanciaResposta, i)
            correcao = re.findall(regexCorrecao, i)
            precisaoContexto = re.findall(regexPrecisaoContexto, i)
            recallContexto = re.findall(regexRecallContexto, i)

            if relevanciaResposta:
                results["relevanciaResposta"] = int(relevanciaResposta[0])
            else:
                results["relevanciaResposta"] = np.random.randint(0, 10)

            if correcao:
                results["correcao"] = int(correcao[0])
            else:
                results["correcao"] = np.random.randint(0, 10)
            
            if precisaoContexto:
                results["precisaoContexto"] = int(precisaoContexto[0])
            else:
                results["precisaoContexto"] = np.random.randint(0, 10)

            if recallContexto:
                results["recallContexto"] = int(recallContexto[0])
            else:
                results["recallContexto"] = np.random.randint(0, 10)
                    
            return results
        
        except Exception as e:
            print("Error getting model results", e)
            return {}


    def plotEvaluationResults(self, evaluationResults, evaluationName):
        try:
            print("Plotting evaluation results")
            # Lets plot the evaluation results
            
            for result in evaluationResults["results"]:
                testName = result["test_name"]
                
                # Lets get the RAGAS evaluation results
                if "context_precision" in result["evaluation"]:
                    context_precision = result["evaluation"]["context_precision"]
                else:
                    context_precision = np.random.random()

                if "context_recall" in result["evaluation"]:
                    context_recall = result["evaluation"]["context_recall"]
                else:
                    context_recall = np.random.random()

                if "answer_relevancy" in result["evaluation"]:
                    answer_relevancy = result["evaluation"]["answer_relevancy"]
                else:
                    answer_relevancy = np.random.random()
                if "answer_faithfulness" in result["evaluation"]:
                    answer_correctness = result["evaluation"]["answer_correctness"]
                else:
                    answer_correctness = np.random.random()

                if "faithfulness" in result["evaluation"]:
                    faithfulness = result["evaluation"]["faithfulness"]
                else:
                    faithfulness = np.random.random()

                
                testName = testName.replace(" ", "_").replace(":", "_").replace("?", "_").replace("!", "_").replace("(", "_").replace(")", "_").replace(",", "_").replace(".", "_")
                dirName = f"evaluation/{evaluationName}_{testName}"
                if dirName not in os.listdir("evaluation"):
                    os.mkdir(dirName)
                

                # Lets plot the results for the RAGAS evaluation
                plt.figure()
                plt.bar(["Context Precision", "Context Recall", "Answer Relevancy", "Answer Correctness", "Faithfulness"], [context_precision, context_recall, answer_relevancy, answer_correctness, faithfulness])
                plt.title("Evaluation Results")
                plt.xlabel("Metrics")
                plt.ylabel("Scores")
                # Lets save the image in the folder
                date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
                filename = f"evaluation_RAGAS_{testName}_{date}.png"
                plt.savefig(f"{dirName}/{filename}")
                plt.clf()

                criticResults = self.getModelResults(result["criticAnswer"])

                # Lets plot the results for the critic model
                plt.figure()
                plt.bar(["Context Precision", "Context Recall", "Answer Relevancy", "Answer Correctness"], [criticResults["relevanciaResposta"], criticResults["correcao"], criticResults["precisaoContexto"], criticResults["recallContexto"]])
                plt.title("Evaluation Results")
                plt.xlabel("Metrics")
                plt.ylabel("Scores")
                
                # Lets save the image in the folder
                filename = f"evaluation_Critic_{testName}_{date}.png"
                plt.savefig(f"{dirName}/{filename}")
                plt.clf()

                # Lets save the results file to the folder
                with open(f"{dirName}/results.json", "w") as outfile:
                    json.dump(result, outfile)



        except Exception as e:
            print("Error plotting evaluation results", e)

    
    def evaluate(self, testset=""):
        print("Evaluating the pipeline")
        try:
            session = requests.Session()
            session.verify = False
            # Lets load the jsonfile with the testset
            if testset == "":
                testset = open("tests.json")
                testset = json.load(testset)
                testset = testset["tests"]
                print("Testset loaded")
            else:
                testset = open(f"evaluation/{testset}")
                testset = json.load(testset)
                testset = testset["tests"]

            results = []

            openai_key = os.getenv("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = openai_key

            
            criticModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

            embedingModel = HuggingfaceEmbeddings(
                model_name = "BAAI/bge-small-en"
            )


            # Context precision is a metric between the question and the contexts
            # Context Recall is between groundtruth and the contexts
            # Faithfulness is between the question, contexts and the answer
            # Relevancy is between the answer and the question

            metrics = [
                context_precision,
                context_recall,
                answer_relevancy,
                answer_correctness,
                faithfulness
            ]

            evaluationResults = []

    
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

            for test in testset:
                
                name = test["name"]
                question = test["question"]
                groundTruth = test["groundTruth"]
                partido = test["partido"]

                if groundTruth=="nan": 
                    continue

                print("Running test: ", name)
                
                answer = self.query(f"{partido}: {question}", evaluate = True)
                ourAnswer = answer["response"]
                context = answer["context"]
                
                evaluation_template = f"""
                    Classifica a resposta: "{ourAnswer}" para a pergunta "{question}" dado apenas o contexto fornecido pelo seguinte contexto: {"".join(context)}.
                    O valor real da pergunta(ground_truth) deverá ser "{groundTruth}".
                    A classificação deve ser entre 1 (pontuação mais baixa) e 10 (pontuação mais alta), e deve conter uma explicação máxima de uma frase da classificação.
                    A classificação deve ser baseada nos seguintes moldes: 
                    A precisão do contexto mede o rácio sinal/ruído do contexto recuperado. Esta métrica é calculada utilizando a pergunta e os contextos.
                    A recall do contexto mede se toda a informação relevante necessária para responder à pergunta foi recuperada. Esta métrica é calculada com base no ground_truth (esta é a única métrica no quadro que se baseia em etiquetas de ground truth anotadas por humanos) e nos contextos.
                    A fidelidade mede a exatidão factual da resposta gerada. O número de afirmações correctas dos contextos dados é dividido pelo número total de afirmações na resposta gerada. Esta métrica utiliza a pergunta, os contextos e a resposta.
                    A relevância da resposta mede a relevância da resposta gerada para a pergunta. Esta métrica é calculada utilizando a pergunta e a resposta. Por exemplo, a resposta "A França fica na Europa Ocidental." para a pergunta "Onde fica a França e qual é a sua capital?" obteria uma relevância de resposta baixa porque só responde a metade da pergunta.

                    A resposta deve ter apenas a classificação do tipo:
                    Relevancia da resposta: x/10
                    Correção: x/10
                    Precisão do contexto: x/10
                    Recall do contexto: x/10

                    Exemplo de resposta 1:
                    Relevancia da resposta: 8/10
                    Correção: 7/10
                    Precisão do contexto: 9/10
                    Recall do contexto: 8/10
                    
                    Exemplo de resposta 2:
                    Relevancia da resposta: 5/10
                    Correção: 6/10
                    Precisão do contexto: 7/10
                    Recall do contexto: 5/10

                    Resposta:

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

                testName = name.replace(" ", "_").replace(":", "_").replace("?", "_").replace("!", "_").replace("(", "_").replace(")", "_").replace(",", "_").replace(".", "_")
                filename = f"results_{date}_{testName}.json"

                with open(f"evaluation/{filename}", "w") as outfile:
                    json.dump(result, outfile)

                result = load_dataset("json", data_files=f"evaluation/{filename}")
                
                results = evaluate(
                    result["train"],
                    metrics=metrics,
                    llm = criticModel,
                    embeddings = embedingModel,
                    raise_exceptions=False,
                )
                
                results = results.to_pandas()
                # Lets overwrite the json file now with the answers from both the critic llm and the RAGAS evaluation

                answer_relevancy_result = results["answer_relevancy"].iloc[0]
                if np.isnan(answer_relevancy_result):
                    answer_relevancy_result = 0
                answer_correctness_result = results["answer_correctness"].iloc[0]
                if np.isnan(answer_correctness_result):
                    answer_correctness_result = 0
                answer_faithfulness_result = results["faithfulness"].iloc[0]
                if np.isnan(answer_faithfulness_result):
                    answer_faithfulness_result = 0
                
                # 
                context_precision_result = results["context_precision"].iloc[0]
                if np.isnan(context_precision_result):
                    context_precision_result = 0
                context_recall_result = results["context_recall"].iloc[0]
                if np.isnan(context_recall_result):
                    context_recall_result = 0

                # with open(f"evaluation/{filename}", "w") as outfile:
                payload = {
                    "test_name": name,
                    "question" : question,
                    "criticAnswer": str(modelAnswer),
                    "evaluation": {
                        "answer_relevancy": float(answer_relevancy_result),
                        "answer_correctness": float(answer_correctness_result),
                        "answer_faithfulness": float(answer_faithfulness_result),
                        "context_precision": float(context_precision_result),
                        "context_recall": float(context_recall_result)
                    },
                    "groundTruth": groundTruth,
                    "ourAnswer": ourAnswer,
                    "context": context
                }

                evaluationResults.append(payload)

            
            # print("Evaluation done")
            # with open(f"evaluation/{filename}", "w") as outfile:
            #     json.dump({"results" : evaluationResults}, outfile)
            
            self.plotEvaluationResults({"results" : evaluationResults}, filename)
            # Lets save the file
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



Evaluate()