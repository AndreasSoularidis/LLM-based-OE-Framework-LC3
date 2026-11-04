import argparse
import json
import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from FileLoaders import FileLoader
from OntologyExtractor import OntologyExtractor
from ChatExtractor import ChatExtractor
from Modes import RAGModes, FrameworkModes
from ModelLoader import ModelLoader
from EmbeddingsLoader import EmbeddingsLoader
from ChainBuilder import Builder
from PromptBuilder import CustomPromptBuilder


load_dotenv()


def parse_args():
  parser = argparse.ArgumentParser(description="Run the LLM4ACOE ontology simulation.")
  parser.add_argument("--provider", choices=["openai", "azure-openai", "anthropic", "google", "nvidia"], default="openai")
  parser.add_argument("--model", default="gpt-4o")
  parser.add_argument("--temperature", type=float, default=0.5)
  parser.add_argument("--max-tokens", type=int, default=None)
  parser.add_argument("--top-p", type=float, default=None)
  parser.add_argument("--model-kwargs", type=str, default=None, help="Optional JSON object of extra model kwargs.")
  parser.add_argument("--embedding-provider", choices=["huggingface", "openai", "azure-openai"], default="huggingface")
  parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
  parser.add_argument("--embedding-fallback-model", default="sentence-transformers/all-MiniLM-L6-v2")
  return parser.parse_args()


ARGS = parse_args()
# Hyperparameters
MODE = FrameworkModes.PARALLEL
PROVIDER = ARGS.provider
MODEL = ARGS.model
TEMPERATURE = ARGS.temperature
EMBEDDING_PROVIDER = ARGS.embedding_provider
EMBEDDING_MODEL = ARGS.embedding_model
EMBEDDING_FALLBACK_MODEL = ARGS.embedding_fallback_model
CHUNK_SIZE = 750
DOMAIN_CHUNK_SIZE = CHUNK_SIZE
CHUNK_OVERLAP = 0
SEARCH_TYPE = 'mmr'
K = 4
RAG_MODE = RAGModes.SAR_OWL_REACT
TYPE = "Test"
ITERATION = 1

# Data sources
DOMAIN_DOCUMENTS = "data/SAR_docs_text"
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
  embeddings = EmbeddingsLoader.load_embeddings(
    provider=EMBEDDING_PROVIDER,
    model=EMBEDDING_MODEL,
    fallback_model=EMBEDDING_FALLBACK_MODEL,
  )
  vector_store = FAISS.from_documents(docs, embeddings)
  return vector_store


def build_model_kwargs(args):
  extra_kwargs = {}
  if args.max_tokens is not None:
    extra_kwargs["max_tokens"] = args.max_tokens
  if args.top_p is not None:
    extra_kwargs["top_p"] = args.top_p
  if args.model_kwargs:
    try:
      parsed = json.loads(args.model_kwargs)
      if not isinstance(parsed, dict):
        raise ValueError("--model-kwargs must be a JSON object")
      extra_kwargs.update(parsed)
    except json.JSONDecodeError as exc:
      raise ValueError(f"Invalid JSON for --model-kwargs: {args.model_kwargs}") from exc
  return extra_kwargs


def start_simulation(question, domain_retriever, owl_retriever, react_retriever):
  template = CustomPromptBuilder.build(RAG_MODE.value, question)
  custom_rag_prompt = PromptTemplate.from_template(template)
  rag_chain = Builder.create_rag_chain(RAG_MODE.value, domain_retriever, owl_retriever, react_retriever, model, custom_rag_prompt)
  response = rag_chain.invoke(question)

  return response


def start_enhanced_simulation(question, round, domain_retriever, owl_retriever, react_retriever, ontology=None):
  template = CustomPromptBuilder.enhanced_build(RAG_MODE.value, question, round, ontology)
  custom_rag_prompt = PromptTemplate.from_template(template)
  rag_chain = Builder.create_enhanced_rag_chain(RAG_MODE.value, round, domain_retriever, owl_retriever, react_retriever, model, custom_rag_prompt)
  
  if ontology == None:
    response = rag_chain.invoke(question)
  else:
    text = "Retrieve the most relevant content that provide detailed instructions for creating expressive and robost ontologies"
    response = rag_chain.invoke(text)
  return response

if __name__ == '__main__':
  print("Initialization. Please wait..\n")
  model_kwargs = build_model_kwargs(ARGS)
  model = ModelLoader.load_llm(PROVIDER, MODEL, TEMPERATURE, model_kwargs=model_kwargs)
  
  react_retriever = owl_retriever = domain_retriever = None

  if RAG_MODE.value == 1 or RAG_MODE.value == 2 or RAG_MODE.value == 3: # ReAct
    react_docs = FileLoader.get_txt_loader(REACT_DOCUMENTS)
    react_docs = split_documents(react_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    react_vector_store = create_vector_store(react_docs)
    react_retriever = react_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k":K})
  
  if RAG_MODE.value == 1 or RAG_MODE.value == 2: # ReAct & Domain Data
    domain_docs = FileLoader.get_txt_loader(DOMAIN_DOCUMENTS)
    domain_docs = split_documents(domain_docs, DOMAIN_CHUNK_SIZE, CHUNK_OVERLAP)
    domain_vector_store = create_vector_store(domain_docs)
    domain_retriever = domain_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k": K})
  
  if RAG_MODE.value == 1: # ReAct & Domain Data & OWL Documentation
    owl_docs = FileLoader.get_web_loader(OWL_DOCUMENTATION)
    owl_docs = split_documents(owl_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    owl_vector_store = create_vector_store(owl_docs)
    owl_retriever = owl_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k":K})

  filename = f"{MODE.name}-{MODEL}-{TEMPERATURE}-{CHUNK_SIZE}-{SEARCH_TYPE}-{K}-{RAG_MODE}-{TYPE}-{ITERATION}"
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
      response = start_simulation(user_input, domain_retriever, owl_retriever, react_retriever)
      chat += response
      print(response)
      ontology_extractor.export_last_to_ttl(response)
    else:
      for step in range(1, 4): 
        response = start_enhanced_simulation(user_input, step, domain_retriever, owl_retriever, react_retriever, ontology)
        chat += response + '\n' 
        print(response)
        ontology = ontology_extractor.export_last_to_ttl(response, step)

  print("Simulation completed!")
  chat_extractor.discussion_extractor(chat)

