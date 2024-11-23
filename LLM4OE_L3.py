from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from FileLoaders import FileLoader
from OntologyExtractor import OntologyExtractor
from ChatExtractor import ChatExtractor
from Modes import RAGModes, FrameworkModes
from ModelLoader import ModelLoader
from ChainBuilder import Builder
from PromptBuilder import CustomPromptBuilder


load_dotenv()
# Hyperparameters
MODE = FrameworkModes.PARALLEL
MODEL = "gpt-4o"
TEMPERATURE = 0.5
CHUNK_SIZE = 750
DOMAIN_CHUNK_SIZE = CHUNK_SIZE
CHUNK_OVERLAP = 0
SEARCH_TYPE = 'mmr'
K = 4
RAG_MODE = RAGModes.SAR_OWL_REACT
TYPE = "Test"
ITERATION = 1

# Data sources
SAR_DOCUMENTS = "data/csv"
# SAR_DOCUMENTS = "data/SAR_docs_text"
OWL_DOCUMENTATION = "https://www.w3.org/2007/OWL/draft/owl2-primer/#Classes.2C_Properties.2C_and_Individuals_.E2.80.93_And_Basic_Modeling_With_Them"
REACT_DOCUMENTS = "data/ReAct-v2"


# Path for results
RESULTS_PATH ="SAR/Level3/Phase_2"

def split_documents(docs, chunk_size, chunk_overlap):
  splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap
  )
  return splitter.split_documents(docs)


def create_vector_store(docs):
  embeddings = OpenAIEmbeddings()
  vector_store = FAISS.from_documents(docs, embeddings)
  return vector_store


def start_simulation(question, sar_retriever, owl_retriever, react_retriever):
  template = CustomPromptBuilder.build(RAG_MODE.value, question)
  custom_rag_prompt = PromptTemplate.from_template(template)
  rag_chain = Builder.create_rag_chain(RAG_MODE.value, sar_retriever, owl_retriever, react_retriever, model, custom_rag_prompt)
  response = rag_chain.invoke(question)

  return response


def start_enhanced_simulation(question, round, sar_retriever, owl_retriever, react_retriever, ontology=None):
  template = CustomPromptBuilder.enhanced_build(RAG_MODE.value, question, round, ontology)
  custom_rag_prompt = PromptTemplate.from_template(template)
  rag_chain = Builder.create_enhanced_rag_chain(RAG_MODE.value, round, sar_retriever, owl_retriever, react_retriever, model, custom_rag_prompt)
  
  if ontology == None:
    response = rag_chain.invoke(question)
  else:
    text = "Retrieve the most relevant content that provide detailed instructions for creating expressive and robost ontologies"
    response = rag_chain.invoke(text)
  return response

if __name__ == '__main__':
  print("Initialization. Please wait..\n")
  
  model = ModelLoader.load_llm(MODEL, TEMPERATURE)
  
  react_retriever = owl_retriever = sar_retriever = None

  if RAG_MODE.value == 1 or RAG_MODE.value == 2 or RAG_MODE.value == 3: # ReAct
    react_docs = FileLoader.get_txt_loader(REACT_DOCUMENTS)
    react_docs = split_documents(react_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    react_vector_store = create_vector_store(react_docs)
    react_retriever = react_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k":K})
  
  if RAG_MODE.value == 1 or RAG_MODE.value == 2: # ReAct & Domain Data
    #sar_docs = FileLoader.get_txt_loader(SAR_DOCUMENTS)
    sar_docs = FileLoader.get_csv_loader(SAR_DOCUMENTS)
    sar_docs = split_documents(sar_docs, DOMAIN_CHUNK_SIZE, CHUNK_OVERLAP)
    sar_vector_store = create_vector_store(sar_docs)
    sar_retriever = sar_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k": K})
  
  if RAG_MODE.value == 1: # ReAct & Domain Data & OWL Documentation
    owl_docs = FileLoader.get_web_loader(OWL_DOCUMENTATION)
    owl_docs = split_documents(owl_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    owl_vector_store = create_vector_store(owl_docs)
    owl_retriever = owl_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k":K})

  filename = f"{MODEL}-{TEMPERATURE}-{CHUNK_SIZE}-{SEARCH_TYPE}-{K}-{RAG_MODE}-{TYPE}-{ITERATION}"
  ontology_extractor = OntologyExtractor(RESULTS_PATH + "/Ontologies", "Ontology-" + filename)
  chat_extractor = ChatExtractor(RESULTS_PATH + "/Discussions", "Chat-" + filename)
  
  ontology = None
  response = None
  chat = ""
  while True:

    user_input = input("You: ")
    
    if user_input.lower().strip() == "exit":
      break
    
    if MODE.name == "PARALLEL":
      response = start_simulation(user_input, sar_retriever, owl_retriever, react_retriever)
      chat += response
      print(response)
      ontology_extractor.export_last_to_ttl(response)
    else:
      for step in range(1, 4): 
        response = start_enhanced_simulation(user_input, step, sar_retriever, owl_retriever, react_retriever, ontology)
        chat += response + '\n' 
        print(response)
        ontology = ontology_extractor.export_last_to_ttl(response, step)

  print("Simulation completed!")
  chat_extractor.discussion_extractor(chat)

