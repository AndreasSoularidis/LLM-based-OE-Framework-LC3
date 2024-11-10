from PromptTempates import CustomPromptTemplates

class CustomPrompt():
    def __init__(self):
        self.prompt = ""
        self.role = ""
        self.user_query = ""
        self.context = ""
        self.section = ""
        self.tail = ""

    def add_role(self, role):
        self.role = role
        return self

    def add_context(self, text):
        self.context += text
        return self

    def add_user_query(self, query):
        self.user_query = query
        return self
    
    def add_section(self, text):
        self.section += text
        return self
    
    def add_tail(self, text):
        self.tail = text
        return self
    
    def build(self):
        self.prompt = self.role + self.context + self.user_query + self.section + self.tail
        return self.prompt
    

class CustomPromptBuilder():
    def build(mode, query):
        builder = CustomPrompt()

        if mode == 1:
            return (builder.add_context(CustomPromptTemplates.context_L3)
                     .add_user_query(query)
                     .add_section(CustomPromptTemplates.react_section)
                     .add_section(CustomPromptTemplates.owl_section)
                     .add_section(CustomPromptTemplates.domain_section)
                     .add_tail(CustomPromptTemplates.tail_section)
                     .build()
           )
        
        if mode == 2:
            return (builder.add_context(CustomPromptTemplates.context_L3)
                     .add_user_query(query)
                     .add_section(CustomPromptTemplates.react_section)
                     .add_section(CustomPromptTemplates.domain_section)
                     .add_tail(CustomPromptTemplates.tail_section)
                     .build()
           )
        
        if mode == 3:
            return (builder.add_context(CustomPromptTemplates.context_L3)
                     .add_user_query(query)
                     .add_section(CustomPromptTemplates.react_section)
                     .add_tail(CustomPromptTemplates.tail_section)
                     .build()
           )
        
        if mode == 4:
            return (builder.add_context(CustomPromptTemplates.context_L3)
                     .add_user_query(query)
                     .add_tail(CustomPromptTemplates.tail_section)
                     .build()
           )
        