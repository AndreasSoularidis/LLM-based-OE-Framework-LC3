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

model = ModelLoader.load_llm(MODEL, TEMPERATURE)

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


def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)


def process_chat(question, sar_retriever, owl_retriever, react_retriever):
    template = CustomPromptBuilder.build(MODE.value, question)
    custom_rag_prompt = PromptTemplate.from_template(template)
    rag_chain = Builder.create_rag_chain(MODE.value, sar_retriever, owl_retriever, react_retriever, model, custom_rag_prompt)

    response = rag_chain.invoke(question)

    return response


if __name__ == '__main__':
  print("Initialization..")
  sar_docs = FileLoader.get_txt_loader(SAR_DOCUMENTS)
  owl_docs = FileLoader.get_web_loader(OWL_DOCUMENTATION)
  react_docs = FileLoader.get_txt_loader(REACT_DOCUMENTS)

  sar_docs = split_documents(sar_docs, CHUNK_SIZE, CHUNK_OVERLAP)
  owl_docs = split_documents(owl_docs, CHUNK_SIZE, CHUNK_OVERLAP)
  react_docs = split_documents(react_docs, CHUNK_SIZE, CHUNK_OVERLAP)

  sar_vector_store = create_vector_store(sar_docs)
  owl_vector_store = create_vector_store(owl_docs)
  react_vector_store = create_vector_store(react_docs)

  sar_retriever = sar_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k": K})
  owl_retriever = owl_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k":K})
  react_retriever = react_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k":K})

  filename = f"{MODEL}-{TEMPERATURE}-{CHUNK_SIZE}-{SEARCH_TYPE}-{K}-{MODE}-{ITERATION}-Test"
  ontology_extractor = OntologyExtractor("SAR/Level3/Phase_2/Ontologies", "Ontology-" + filename)
  chat_extractor = ChatExtractor("SAR/Level3/Phase_2/Discussions", "Chat-" + filename)
  
  while True:

    user_input = input("You: ")
    
    if user_input.lower().strip() == "exit":
      break
    
    match MODE.value:
      case 1:
        response = process_chat(user_input, sar_retriever, owl_retriever, react_retriever) 
      case 2:
        response = process_chat(user_input, sar_retriever, None, react_retriever)
      case 3:
        response = process_chat(user_input, None, None, react_retriever)
      case 4:
        response = process_chat(user_input, None, None, None)
      case 5:
        response = process_chat(user_input, sar_retriever, None, None)
      case _:
        print("Error: No mode was selected")

    print(response)

    ontology_extractor.export_to_ttl(response)
  chat_extractor.discussion_extractor(response)

