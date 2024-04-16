from pinecone import Pinecone, PodSpec

from llama_index.vector_stores.pinecone import PineconeVectorStore

from llama_index.llms.together import TogetherLLM

from dotenv import load_dotenv
import os
import time


class DBController():

    def __init__(self):
        #Lets verify if the database is up and running
        load_dotenv()
        self.db = self.initDatabase()
        self.dropDB("MDRag")
        self.createIndex("MDRag")
    

    def dropDB(self, index_name):
        if index_name in self.db.list_indexes().names():
            self.db.delete_index(index_name)
            print("Index deleted")

    def createIndex(self, index_name):
        try:
            # Verify if index is already created
            existed = False
            if index_name not in self.db.list_indexes().names():
                # 384 is the default dimension for the embedding model
                # We will use dotproduct since we want to tell how similar two vectors are
                self.db.create_index(name=index_name,
                                    dimension=384,
                                    metric = "cosine",
                                    spec=PodSpec(
                                        environment="gcp-starter"
                                    )
                    )
                existed=True
            while not self.db.describe_index(index_name).status["ready"]:
                print("Waiting for the index to be created")
                time.sleep(1)

            self.index = self.db.Index(index_name)
            self.vector_store = PineconeVectorStore(self.index)
            print("Index created")
            return existed

        except Exception as e:
            print("Error creating index", e)


    def insert(self, data):
        try:
            
            self.index.upsert([data])

        except Exception as e:
            print("Error inserting data", e)

    def runQuery(self, query):
        try:
            
            result = self.index.query(
                vector=query,
                top_k=5,
                include_values=True
            )


            results = result["matches"]
            # print(results)

            docs = {}
            for result in results:
                text = self.index.query(id=result["id"], top_k=1, include_metadata=True)
                # # biggerWindow = ""
                # try:
                #     previous_text = self.index.query(id=result["id"]-1, top_k=1, include_metadata=True)["matches"][0]["metadata"]["text"]
                #     next_text = self.index.query(id=result["id"]+1, top_k=1, include_metadata=True)["matches"][0]["metadata"]["text"]
                #     text = previous_text + text["matches"][0]["metadata"]["text"] + next_text
                # except:
                #     text = text["matches"][0]["metadata"]["text"]
                

                # # Print result
                score = result["score"]
                text = text["matches"][0]["metadata"]["texto"]
                print("Score:", score)
                print("Query response:", text)
                if score not in docs:
                    docs[score] = [text]
                else:
                    docs[score].append(text)

            return docs

        except Exception as e:
            print("Error querying", e)



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


    def initDatabase(self):
        try:
            api_key = os.getenv("PINECONE_API_KEY")
            print("Pinecone API Key", api_key)
            pinecone = Pinecone(api_key=api_key)
            print("Database initialized")
            return pinecone

        except Exception as e:
            print("Error initializing database", e)
            return None
