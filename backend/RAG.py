#Utils
from dotenv import load_dotenv
import os

#RAG pipeline

# Alter from the llama index to the https://docs.together.ai/docs/quickstart
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import uuid

# RAG document ingestion pipeline
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
# Text splitter to split the documents into chunks
from llama_index.core.node_parser import SentenceSplitter

#Database
from DBController import DBController

import numpy as np
import spacy


#TODO Make run requirements and install spacy python3 -m spacy download en_core_web_sm
#TODO export TOKENIZERS_PARALLELISM=true

class RAG():
    def __init__(self):
        load_dotenv()
        self.DBController = DBController()
        self.llm = self.loadModel()
        self.embedingModel = self.loadEmbeddingModel()
        # self.dataIngestion()
        
        
    def query(self, query):
        try:
            options = {"partidos": []}
            
            #query.split(" ")
            # TODO: identificar a presença de partidos e temáticas nas queries de forma a correr cada query à base de dados com base nesses partidos
            # Exemplo: Quais as medidas do ps e da ad para a economia?
            # Partidos: PS, AD
            # Temáticas: Economia
            # Query na base de dados ao PS sobre a economia
            # Query na base de dados à AD sobre a economia
            # Passar tudo como contexto
            # Finalizar isto e ver se está correto
            partidos = ["Chega", "AD", "Aliança democrática", "Bloco de esquerda", "CDS", "CDU", "Iniciativa Liberal"]

            embededQuery = self.generateEmbeddings(query)
            # extraContext = self.DBController.runQuery(embededQuery, options)
            extraContext = self.DBController.runQuery(embededQuery)
        
            minimumConfidence = 0.80
            contextAdd = ""
            for confidenceLevel in extraContext:
                if confidenceLevel > minimumConfidence:
                    for context in extraContext[confidenceLevel]:
                        contextAdd += context + "\n"
            if extraContext == "" or min(extraContext) < minimumConfidence:
                response = "Não encontrei nada sobre isso."
            else:
                # Detetar partido para inserir no contexto
                query = f"""
                            Com base única e exclusivamente no seguinte contexto do plano eleitoral para 2024:\n{contextAdd}
                            Dá me a resposta à seguinte questão em Português de Portugal!\n
                            Pergunta: {query}"""
                
                response = self.llm.complete(query)
            
            return str(response)
                
        
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
            self.nlp = spacy.load('en_core_web_sm')

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
                elif cluster_len > 3000:
                    threshold = 0.6
                    sents_div, vecs_div = self.process(cluster_txt)
                    reclusters = self.cluster_text(sents_div, vecs_div, threshold)
                    
                    for subcluster in reclusters:
                        div_txt = ' '.join([sents_div[i].text for i in subcluster])
                        div_len = len(div_txt)
                        
                        if div_len < 60 or div_len > 3000:
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
            for party in os.listdir("data"):
                for doc in os.listdir(f"data/{party}"):
                
                    docName = doc.split(".")[0]
                    docs.append(doc)
                    # document = loader.load(file_path=f"./data/{doc}")

                    data = open(f"./data/{party}/{doc}")
                    data = data.readlines()
                    data = "\n".join(data)

                    chunkedData = self.chunkFile(file=data, tokensPerChunk=300, overlap=50)
                    # data = loader.load(file_path=f"./data/{party}/{doc}")
                    
                    # In the chunker limit the chunking token size to the number of tokens that the embeder can process

                    print("Data chunked")
                    for idx in range(len(chunkedData)):
                        text_chunk = chunkedData[idx]
                        id=f"{party}_{docName}_{idx}"
                    

                        nodeEmbeding = self.generateEmbeddings(text_chunk)
                        # print(f"Embeding size: {len(nodeEmbeding)}")
                        # TODO: Convém indexar o link do documento original para meter nos metadados
                        # TODO: Fix no id para ir buscar contexto adicional 
                        payload = {
                            "id": str(uuid.uuid4()),
                            "values": nodeEmbeding,
                            "metadata": {
                                "partido": party,
                                "assunto" : docName,
                                "texto" : text_chunk,
                                "source": "teste"
                            }
                        }
                        self.DBController.insert(payload)
                
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



    def loadModel(self):
        try: 
            togetherai_api_key = os.getenv("TogetherAI_API_KEY")
            print("TogetherAI_API_KEY", togetherai_api_key)
            llm = TogetherLLM(
                model = "mistralai/Mixtral-8x7B-Instruct-v0.1",
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
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
            print("Embedding model loaded")
            return embed_model
        except Exception as e:
            print("Error loading embedding model", e)
            return None
        

    def compareSolution(self, query):
        try:
            ourAnswer = self.query(query)
        
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
                print(f"LLM answer: {self.llm.complete(option)}")

                print(f"Our answer: {self.DBController.runQuery(option)}")
                option = input("What is the query you want to make?")
            except Exception as e:
                print("Error testing queries", e)

RAG()
