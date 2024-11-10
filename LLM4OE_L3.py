from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from FileLoaders import FileLoader
from OntologyExtractor import OntologyExtractor
from ChatExtractor import ChatExtractor
from Modes import RAGModes
from ModelLoader import ModelLoader
from ChainBuilder import Builder
from PromptBuilder import CustomPromptBuilder


load_dotenv()
# Hyperparameters
MODEL = "gpt-4o"
TEMPERATURE = 0.5
CHUNK_SIZE = 750
CHUNK_OVERLAP = 0
SEARCH_TYPE = 'mmr'
K = 4
MODE = RAGModes.BARE_LLM
ITERATION = 13

# Data sources
SAR_DOCUMENTS = "data/SAR_docs_text"
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
    template = CustomPromptBuilder.build(MODE.value, question)
    custom_rag_prompt = PromptTemplate.from_template(template)
    rag_chain = Builder.create_rag_chain(MODE.value, sar_retriever, owl_retriever, react_retriever, model, custom_rag_prompt)

    response = rag_chain.invoke(question)

    return response


if __name__ == '__main__':
  print("Initialization. Please wait..\n")
  
  model = ModelLoader.load_llm(MODEL, TEMPERATURE)
  
  react_retriever = owl_retriever = sar_retriever = None

  if MODE.value == 1 or MODE.value == 2 or MODE.value == 3: # ReAct
    react_docs = FileLoader.get_txt_loader(REACT_DOCUMENTS)
    react_docs = split_documents(react_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    react_vector_store = create_vector_store(react_docs)
    react_retriever = react_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k":K})
  
  if MODE.value == 1 or MODE.value == 2: # ReAct & Domain Data
    sar_docs = FileLoader.get_txt_loader(SAR_DOCUMENTS)
    sar_docs = split_documents(sar_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    sar_vector_store = create_vector_store(sar_docs)
    sar_retriever = sar_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k": K})
  
  if MODE.value == 1: # ReAct & Domain Data & OWL Documentation
    owl_docs = FileLoader.get_web_loader(OWL_DOCUMENTATION)
    owl_docs = split_documents(owl_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    owl_vector_store = create_vector_store(owl_docs)
    owl_retriever = owl_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k":K})

  filename = f"{MODEL}-{TEMPERATURE}-{CHUNK_SIZE}-{SEARCH_TYPE}-{K}-{MODE}-{ITERATION}-Test"
  ontology_extractor = OntologyExtractor(RESULTS_PATH + "/Ontologies", "Ontology-" + filename)
  chat_extractor = ChatExtractor(RESULTS_PATH + "/Discussions", "Chat-" + filename)
  
  response = None
  while True:

    user_input = input("You: ")
    
    if user_input.lower().strip() == "exit":
      break

    response = start_simulation(user_input, sar_retriever, owl_retriever, react_retriever) 
    print(response)

    ontology_extractor.export_to_ttl(response)

  print("Simulation completed!")
  chat_extractor.discussion_extractor(response)

