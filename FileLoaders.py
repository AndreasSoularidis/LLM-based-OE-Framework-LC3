from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import bs4
import glob

class FileLoader:
  
  @staticmethod
  def get_web_loader(url):
    web_loader = WebBaseLoader(
      url,
      bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
          class_=("turtle")
      )
    )                            )
    # web_loader.requests_kwargs = {'verify':False}
    return web_loader.load()
  
  
  @staticmethod
  def get_pdf_loader(folder_name):
    files = glob.glob(f"{folder_name}/*.pdf")
    docs = []
    for file in files:
      text_loader = PyMuPDFLoader(file)
      docs.append(text_loader.load()[0])
    return docs
  

  @staticmethod
  def get_csv_loader(folder_name, delimiter=",", quotechar='"', field_names = []):
    files = glob.glob(f"{folder_name}/*.csv")

    documents = []
    docs = []
    for file in files:
      if len(field_names) == 0:
        csv_loader = CSVLoader(
        file_path=file,
        csv_args={
            "delimiter": delimiter,
            "quotechar": quotechar,
        })
      else:
        csv_loader = CSVLoader(
        file_path=file,
        csv_args={
            "delimiter": delimiter,
            "quotechar": quotechar,
            "fieldnames": field_names
        })
      data = csv_loader.load()
      documents.append(data)
    
    for document in documents:
      for line in document:
        docs.append(line)
    return docs
    
  
  @staticmethod
  def get_txt_loader(folder_name):
    text_loader = DirectoryLoader(f".\\{folder_name}", glob="**\\*.txt", loader_cls=TextLoader, show_progress=True, silent_errors=True)
    docs = text_loader.load()
    return docs
  