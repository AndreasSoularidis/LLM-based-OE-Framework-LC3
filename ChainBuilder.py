from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RagChain:
    def __init__(self, sar_retriever, owl_retriever, react_retriever, model, custom_rag_prompt):
        self.sar_retriever = sar_retriever
        self.owl_retriever = owl_retriever
        self.react_retriever = react_retriever
        self.model = model
        self.custom_rag_prompt = custom_rag_prompt
        self.chain = {}

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def add_sar_context(self):
        self.chain["sar_context"] = self.sar_retriever | self.format_docs
        return self

    def add_owl_context(self):
        self.chain["owl_context"] = self.owl_retriever | self.format_docs
        return self

    def add_react_context(self):
        self.chain["react_context"] = self.react_retriever | self.format_docs
        return self

    def add_question(self):
        self.chain["question"] = RunnablePassthrough()
        return self

    def add_custom_rag_prompt(self):
        self.chain = self.chain | self.custom_rag_prompt
        return self 

    def add_model(self):
        self.chain = self.chain | self.model
        return self

    def add_output_parser(self):
        self.chain = self.chain | StrOutputParser()
        return self

    def build(self):
        return self.chain


class Builder():
    def create_rag_chain(mode, sar_retriver, owl_retriever, react_retriver, model, custom_rag_prompt):
        builder = RagChain(sar_retriver, owl_retriever, react_retriver, model, custom_rag_prompt)
        
        if mode == 1:
            rag_chain = (
                builder.add_sar_context().add_question()
                    .add_owl_context().add_question()
                    .add_react_context().add_question()
                    .add_custom_rag_prompt()
                    .add_model()
                    .add_output_parser()
                    .build()
            )
        elif mode == 2:
            rag_chain = (
                builder.add_sar_context().add_question()
                    .add_react_context().add_question()
                    .add_custom_rag_prompt()
                    .add_model()
                    .add_output_parser()
                    .build()
            )
        elif mode == 3:
            rag_chain = (
                builder.add_react_context().add_question()
                    .add_custom_rag_prompt()
                    .add_model()
                    .add_output_parser()
                    .build()
            )
        elif mode == 5:
            rag_chain = (
                builder.add_sar_context().add_question()
                    .add_custom_rag_prompt()
                    .add_model()
                    .add_output_parser()
                    .build()
            )
        else:
            rag_chain = (
                builder.add_custom_rag_prompt()
                    .add_model()
                    .add_output_parser()
                    .build()
            )

        return rag_chain
