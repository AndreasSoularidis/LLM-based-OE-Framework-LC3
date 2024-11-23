from PromptTempates import CustomPromptTemplates

class CustomPrompt():
    def __init__(self):
        self.prompt = ""

    def add_role(self, role):
        self.prompt += role
        return self

    def add_context(self, text):
        self.prompt += text
        return self

    def add_user_query(self, query):
        self.prompt += query
        return self
    
    def add_section(self, text):
        self.prompt += text
        return self
    
    def add_tail(self, text):
        self.prompt += text
        return self
    
    def build(self):
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
                     .add_tail(CustomPromptTemplates.main_tail)
                     .build()
           )
        
        if mode == 2:
            return (builder.add_context(CustomPromptTemplates.context_L3)
                     .add_user_query(query)
                     .add_section(CustomPromptTemplates.react_section)
                     .add_section(CustomPromptTemplates.domain_section)
                     .add_tail(CustomPromptTemplates.main_tail)
                     .build()
           )
        
        if mode == 3:
            return (builder.add_context(CustomPromptTemplates.context_L3)
                     .add_user_query(query)
                     .add_section(CustomPromptTemplates.react_section)
                     .add_tail(CustomPromptTemplates.main_tail)
                     .build()
           )
        
        if mode == 4:
            return (builder.add_context(CustomPromptTemplates.context_L3)
                     .add_user_query(query)
                     .add_tail(CustomPromptTemplates.main_tail)
                     .build()
           )
        

    def enhanced_build(mode, query, round, ontology):
        builder = CustomPrompt()
        if round == 1:
            return (builder.add_context(CustomPromptTemplates.context_L3)
                .add_user_query(query)
                .add_section(CustomPromptTemplates.domain_section)
                .add_tail(CustomPromptTemplates.main_tail)
                .build()
            )
        if round == 2:
            return (builder.add_context(CustomPromptTemplates.context_L3)
                    .add_section("The aim of the ontology is to model all the necessary concepts and their relationships for Search and Rescue (SAR) missions. The scope of the ontology is wildfire incidents. ")
                    .add_section("The generated ontology is the following.\nSTART OF ONTOLOGY\n")
                    .add_section(ontology)
                    .add_section("The generated ontology is the following.\n END OF ONTOLOGY")
                    .add_section(CustomPromptTemplates.owl_section)
                    .add_tail(CustomPromptTemplates.secondary_tail)
                    .build()
            )
        if round == 3:
            return (builder.add_context(CustomPromptTemplates.context_L3)
                    .add_section("The aim of the ontology is to model all the necessary concepts and their relationships for Search and Rescue (SAR) missions. The scope of the ontology is wildfire incidents. ")
                    .add_section("The generated ontology is the following.\nSTART OF ONTOLOGY\n")
                    .add_section(ontology)
                    .add_section("The generated ontology is the following.\nEND OF ONTOLOGY")
                    .add_section(CustomPromptTemplates.react_context)
                    .add_section(CustomPromptTemplates.react_section)
                    .add_tail(CustomPromptTemplates.secondary_tail)
                    .build()
            )


        