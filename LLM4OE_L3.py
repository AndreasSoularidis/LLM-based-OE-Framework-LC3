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


load_dotenv()
# Hyperparameters
MODEL = "gpt-4o"
TEMPERATURE = 0.5
CHUNK_SIZE = 750
CHUNK_OVERLAP = 0
SEARCH_TYPE = 'mmr'
K = 4
MODE = RAGModes.SAR_REACT
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
    template = """
        Create three instances of yourself playing three different roles in the ontology engineering process 
        based on the HCOME collaborative ontology engineering methodology.
        
        The three roles are the knowledge engineer, the domain expert and the knowledge worker. 
        These three roles work together to create an ontology. The Knowledge Engineer is responsible 
        for the requirements specification, conceptualisation and generation of the ontology. The Domain 
        Expert is an experienced person and provides the requirements for the 
        ontology, terminology, definitions of terms, domain specific explanations of terms and his 
        experience in general. The Knowledge Worker is the user of the ontology 
        and actively participates in the ontology engineering process. The above roles should express 
        their deep knowledge during the conversation. Their aim is to play all three roles, simulating 
        the HCOME methodology. The above mentioned roles will interact with each other, asking and 
        answering questions until a valid and comprehensive ontology is created, which covers all 
        the defined requirements below.
    """
   

    if react_retriever or owl_retriever or sar_retriever:
      template += question +"\n"
      template += """
        During the discussion and design of the ontology you should consider the following additional content. 
      """
    else:
      template += """{question}"""

    if react_retriever:
      template += """  
      You should follow the above way of thinking-acting-observing, but BEHIND THE SCENES and WITHOUT showing this thinking chain during the discussion and the ontology generation process, as given below
      START OF REACT

      {react_context}
      
      END OF REACT 
    """
      
    if owl_retriever:
      template += """
      You should also use the some of the following OWL axioms to make the ontology as expressive as possible. Use ONLY the ontology axioms given in the examples and not the data presented.
      You do not need to use all of them, but only the necessary axioms to create a WELL CONNECTED and EXPRESSIVE ontology.
      START OF OWL DOCUMENTATION
      
      {owl_context} 

      END OF OWL DOCUMENTATION
    """

    if sar_retriever:
      template += """
      The following data describe real Search and Rescue (SAR) missions. You should EXTRACT related concepts (CLASSES
      and OBJECT PROPERTIES) and add them to the generated ontology. Try to extract as much relevant classes and properies as possible.
      START OF DOMAIN DATA
      
      {sar_context} 

      END OF DOMAIN DATA
      """
    
    template += """ 
      The iterative discussion stops when the generated ontology answers all the given competency questions and covers all the requirements of the ontology. 
      Thus create as many classes and properties as possible.
      Feel free to use domain knowledge to extend the ontology with classes and properties to make it as comprehensive as possible. 
      DO NOT STOP until cover all the given requirements.
      Present the iterative discussion and the generated ontology in Turtle (TTL) format WITHOUT individuals. 
    """
        
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

