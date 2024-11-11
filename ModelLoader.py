from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA

class ModelLoader:
    @staticmethod
    def load_llm(llm_name, temperature):
        if llm_name ==  'gpt-4o':
            return ChatOpenAI(model = "gpt-4o-2024-05-13", temperature = temperature)
        if llm_name == 'claude':        
            return ChatAnthropic(model ='claude-3-5-sonnet-20241022', temperature = temperature)
        if llm_name == 'gemini':
            return ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=temperature)
        if llm_name == 'nvidia':
            return ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1", temperature=temperature)

        