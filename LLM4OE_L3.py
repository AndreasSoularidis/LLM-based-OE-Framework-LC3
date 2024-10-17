from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from FileLoaders import FileLoader
from OntologyExtractor import OntologyExtractor
from ChatExtractor import ChatExtractor
from Modes import RAGModes


load_dotenv()
# gpt-4o-2024-08-06
# Hyperparameters
MODEL = "gpt-4o-2024-05-13"
TEMPERATURE = 1.0
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
SEARCH_TYPE = 'similarity'
K = 4
MODE = RAGModes.DOCS_OWL_REACT
ITERATION = 17

model = ChatOpenAI(
  model = MODEL,
  temperature = TEMPERATURE
)


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
      The following data describe real Search and Rescue (SAR) missions. You should extract related concepts (classes
      and properties) and add them to the generated ontology.
      START OF DOMAIN DATA
      
      {sar_context} 

      END OF DOMAIN DATA
      """
    
    template += """ 
      The iterative discussion stops when the generated ontology answers all the given competency questions and covers all the requirements of the ontology. 
      Thus create as many classes and properties as possible.
      Feel free to use domain knowledge to extend the ontology with classes and properties to make it as comprehensive as possible. 
      Present the iterative discussion and the generated ontology in Turtle (TTL) format WITHOUT individuals. 
    """
        
    custom_rag_prompt = PromptTemplate.from_template(template)

    if sar_retriever and owl_retriever and react_retriever:
      rag_chain = (
        { "sar_context": sar_retriever | format_docs, "question": RunnablePassthrough()}
        | { "owl_context": owl_retriever | format_docs, "question": RunnablePassthrough()}
        | {"react_context": react_retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | model
        | StrOutputParser()
      )
    elif owl_retriever and react_retriever:
      rag_chain = (
        { "owl_context": owl_retriever | format_docs, "question": RunnablePassthrough()}
        | {"react_context": react_retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | model
        | StrOutputParser()
      )
    elif react_retriever:
      rag_chain = (
        {"react_context": react_retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | model
        | StrOutputParser()
      )
    else:
      rag_chain = (
        custom_rag_prompt
        | model
        | StrOutputParser()
      )

    response = rag_chain.invoke(question)

    return response


if __name__ == '__main__':
  print("Initialization..")
  sar_docs = FileLoader.get_pdf_loader("data/Refined")
  owl_docs = FileLoader.get_web_loader("https://www.w3.org/2007/OWL/draft/owl2-primer/#Classes.2C_Properties.2C_and_Individuals_.E2.80.93_And_Basic_Modeling_With_Them")
  react_docs = FileLoader.get_txt_loader("data")

  sar_docs = split_documents(sar_docs, CHUNK_SIZE, CHUNK_OVERLAP)
  owl_docs = split_documents(owl_docs, CHUNK_SIZE, CHUNK_OVERLAP)
  react_docs = split_documents(react_docs, CHUNK_SIZE, CHUNK_OVERLAP)

  sar_vector_store = create_vector_store(sar_docs)
  owl_vector_store = create_vector_store(owl_docs)
  react_vector_store = create_vector_store(react_docs)

  sar_retriever = sar_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k": K})
  owl_retriever = owl_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k":K})
  react_retriever = react_vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k":K})

  # ensemble_retriever = EnsembleRetriever(retrievers=[sar_retriever, owl_retriever, react_retriever])


  filename = f"{MODEL}-{TEMPERATURE}-{CHUNK_SIZE}-{SEARCH_TYPE}-{K}-{MODE}-{ITERATION}"
  ontology_extractor = OntologyExtractor("SAR/Level3/Phase_2/Ontologies", "Ontology-" + filename)
  chat_extractor = ChatExtractor("SAR/Level3/Phase_2/Discussions", "Chat-" + filename)
  
  while True:

    user_input = input("You: ")
    
    if user_input.lower().strip() == "exit":
      break

    response = process_chat(user_input, sar_retriever, owl_retriever, react_retriever) 

    print(response)

    ontology_extractor.export_to_ttl(response)
  chat_extractor.discussion_extractor(response)

