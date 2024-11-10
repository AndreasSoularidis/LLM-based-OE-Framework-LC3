class CustomPromptTemplates:

  RAG_template = """
  Act as an ontology engineer. Use the following content (examples, part of the ontology, and common mistakes) as a guide to translate the natural language rule you are given into a SWRL rule. 
    EXAMPLES

    Example 1
    Rule in natural language: IF a Person Has a Sibling who Is a Man, THEN the Person Has a Brother.
    SWRL rule: Person(?p) ^ hasSibling(?p, ?s) ^ Person(?s) ^ IsMan(?s) -> hasBrother(?p, ?s)

    Example 2
    Rule in natural language: IF a Person Has a Phone Number that starts with '+', THEN the Person Has an International Phone Number.
    SWRL rule: Person(?p) ^ PhoneNumber(?number) ^ hasPhoneNumber(?p, ?number) ^ swrlb:startsWith(?number, "+") -> hasInternationalPhoneNumber(?p, true)

    Example 3
    Rule in natural language:  IF a Person Has an Age greater than 17, THEN the Person is an Adult and Has IdCard.
    SWRL rule: Person(?p) ^ hasAge(?p, ?age) ^ Age(?age) ^ swrlb:greaterThan(?age, 17) ^ IdCard(?id) -> Adult(?p) ^ hasIdCard(?p, ?id)

    Example 4
    Rule in natural language:  IF a Publication Has two different Authors, THEN those Authors have Cooperated With each other.
    SWRL rule: Publication(?p) ^ Author(?y) ^ Author(?z) ^ hasAuthor(?p, ?y) ^ hasAuthor(?p, ?z) ^ differentFrom(?y, ?z) -> cooperatedWith(?y, ?z)

    Example 5
    Rule in natural language: IF a Researcher Has Published more than 10 papers in a Top-Tier Conference, THEN the Researcher is a Prominent Researcher and Has a High Citation Impact.
    SWRL rule: Researcher(?r) ^ hasPublished(?r, ?p) ^ Conference(?p) ^ topTierConference(?p) ^ swrlb:greaterThan(count(?p), 10) -> ProminentResearcher(?r) ^ hasHighCitationImpact(?r)

    END OF EXAMPLES

    PART OF ONTOLOGY 

    {context}

    END OF ONTOLOGY

    COMMON GUIDELINES

    Please try to implement the following guidelines:
    1: Variables e.g., (?something) used in Conclusions must be present in Conditions. So create as many variables as needed in Conditions. 
    2: Each atom must have exactly one variable if it represents class.
    3: Each atom must have exactly two variables if it represents object property.
    4: Each concept must be represented by a class and must be present in Conditions.

    END OF COMMON GUIDELINES

    The rule in natural language is the following: {question}
    Give only the final SWRL rule
  """

  ontology_template = """
  Act as an ontology engineer. Use the following content (part of the ontology) as a guide to translate the natural language rule you are given into a SWRL rule. 
    PART OF ONTOLOGY 

    {context}

    END OF ONTOLOGY

    The rule in natural language is the following: {question}
    Give only the final SWRL rule
  """

  examples_template = """  Act as an ontology engineer. Use the following content (examples and common mistakes) as a guide to translate the natural language rule you are given into a SWRL rule.
    EXAMPLES

    Example 1
    Rule in natural language: If a person has a sibling who is a man, then the person has a brother.
    SWRL rule: Person(?p) ^ hasSibling(?p, ?s) ^ Person(?s) ^ IsMan(?s) -> hasBrother(?p, ?s)

    Example 2
    Rule in natural language: If a person has a phone number that starts with '+', then the person has an international phone number.
    SWRL rule: Person(?p) ^ PhoneNumber(?number) ^ hasPhoneNumber(?p, ?number) ^ swrlb:startsWith(?number, "+") -> hasInternationalPhoneNumber(?p, true)

    Example 3
    Rule in natural language:  If a person has an age greater than 17, then the person is an adult.
    SWRL rule: Person(?p) ^ hasAge(?p, ?age) ^ Age(?age) ^ swrlb:greaterThan(?age, 17) -> Adult(?p)

    Example 4
    Rule in natural language:  If a publication has two different authors, then those authors have cooperated with each other.
    SWRL rule: Publication(?p) ^ Author(?y) ^ Author(?z) ^ hasAuthor(?p, ?y) ^ hasAuthor(?p, ?z) ^ differentFrom(?y, ?z) -> cooperatedWith(?y, ?z)

    Example 5
    Rule in natural language: If a researcher has published more than 10 papers in a top-tier conference, then the researcher is a prominent researcher and has a high citation impact.
    SWRL rule: Researcher(?r) ^ hasPublished(?r, ?p) ^ Conference(?p) ^ topTierConference(?p) ^ swrlb:greaterThan(count(?p), 10) -> ProminentResearcher(?r) ^ hasHighCitationImpact(?r)

    END OF EXAMPLES

    COMMON MISTAKES

    Please avoid the following common mistakes:
    Mistake 1: Variables (e.g., ?something) used in Conclusions are not present in Conditions. So create as many atoms as needed.
    Mistake 2: Each atom must have exactly one variable if it represents class.
    Mistake 3: Each atom must have exactly two variables if it represents object property.
    Mistake 4: Each concept must be represented by a class.

    END OF COMMON MISTAKES

    The rule in natural language is the following: {question}
    Give only the final SWRL rule
  """
  
  simple_template = """  Act as an ontology engineer. Translate the natural language rule you are given into a SWRL rule.
    The rule in natural language is the following: {question}
    Give only the final SWRL rule
  """

  context_L3 = """
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

  react_section = """ 
    \nDuring the discussion and design of the ontology you should consider the following additional content. 
    You should follow the above way of thinking-acting-observing, but BEHIND THE SCENES and WITHOUT showing this thinking chain during the discussion and the ontology generation process, as given below
    START OF REACT

    {react_context}
    
    END OF REACT 
  """

  owl_section = """
    You should also use the some of the following OWL axioms to make the ontology as expressive as possible. Use ONLY the ontology axioms given in the examples and not the data presented.
    You do not need to use all of them, but only the necessary axioms to create a WELL CONNECTED and EXPRESSIVE ontology.
    START OF OWL DOCUMENTATION
    
    {owl_context} 

    END OF OWL DOCUMENTATION
  """

  domain_section = """
    The following data describe real Search and Rescue (SAR) missions. You should EXTRACT related concepts (CLASSES
    and OBJECT PROPERTIES) and add them to the generated ontology. Try to extract as much relevant classes and properies as possible.
    START OF DOMAIN DATA
    
    {sar_context} 

    END OF DOMAIN DATA
  """

  tail_section = """ 
    The iterative discussion stops when the generated ontology answers all the given competency questions and covers all the requirements of the ontology. 
    Thus create as many classes and properties as possible.
    Feel free to use domain knowledge to extend the ontology with classes and properties to make it as comprehensive as possible. 
    DO NOT STOP until cover all the given requirements.
    Present the iterative discussion and the generated ontology in Turtle (TTL) format WITHOUT individuals. 
  """


  @staticmethod
  def basic_nl2swrl_prompt():
    header = "Act as an Ontology Engineer, translate the rule from natural language to SWRL. The rule in natural language is the following:\n" 
    easy_rule = """IF gas is recognized at coordinates AND fire is recognized at these coordinates AND agents is nearby to these coordinates THEN send alert to these agents AND agents abort mission."""
    medium_rule = """IF a person is recognized at coordinates AND has heart rate less than 30 pulses per minute AND the location in these coordinates is safe THEN send to coordinates a search and rescue team."""
    hard_rule = """IF house on fire is recognized at coordinates AND person has voice sign 'help' THEN send alert to the fire station AND send To coordinates a search and rescue team  AND send To coordinates a flying drone agent AND send To coordinates a spot robot agent."""
    footer = "\n\nReturn only the SWRL rule.\n"
    
    prompt1 = header + easy_rule + footer
    prompt2 = header + medium_rule + footer
    prompt3 = header + hard_rule + footer

    return [prompt1, prompt2, prompt3]
   

  @staticmethod
  def capital_letters_nl2swrl_prompt():
    header = "Act as an Ontology Engineer, translate the rule from natural language to SWRL. The rule in natural language is the following:\n" 
    easy_rule = """IF Gas is Recognized At Coordinates AND Fire is Recognized At these Coordinates AND Agents is Nearby to these Coordinates THEN send Alert to these Agents AND Agents abort Mission."""
    medium_rule = """IF a Person is Recognized At Coordinates AND has Heart Rate less than 30 pulses per minute AND the Location in these Coordinates is Safe THEN send To Coordinates a Search and Rescue Team."""
    hard_rule = """IF House on Fire is Recognized At Coordinates AND Person has Voice Sign 'help' THEN send Alert to the Fire Station AND send To Coordinates a Search and Rescue Team  AND send To Coordinates a Flying Drone Agent AND send To Coordinates a Spot Robot Agent."""
    footer = "\n\nReturn only the SWRL rule.\n"
    
    prompt1 = header + easy_rule + footer
    prompt2 = header + medium_rule + footer
    prompt3 = header + hard_rule + footer

    return [prompt1, prompt2, prompt3]
  

  @staticmethod
  def examples_nl2swrl_prompt():
    header = "Act as an Ontology Engineer, translate the rule from natural language to SWRL like the examples below.\n" 
    context = """
      Example 1: 
        Natural Language Rule: "If a Person Has Sibling (?s) AND Person(?s) is Man (?s) THEN Has Brother (?s)" 
        SWRL rule: Person(?p) ^ hasSibling(?p, ?s) ^ Person(?s) ^ Man(?s) -> hasBrother(?p, ?s)

      Example 2: 
        Natural Language Rule: "If a Driver Has Age greater than 27 Then his Car Is Fully Insurable" 
        SWRL rule: Driver(?d) ^ hasAge(?d, ?age) ^ swrlb:greaterThan(?age, 25) ^ Car(?car) -> isFullyInsurable(?car, true)\n
      
      The rule in natural language is the following: \n
    """
    footer = "\n\nReturn only the SWRL rule.\n"

    easy_rule = """IF gas is recognized at coordinates AND fire is recognized at these coordinates AND agents is nearby to these coordinates THEN send alert to these agents AND agents abort mission."""
    medium_rule = """IF a person is recognized at coordinates AND has heart rate less than 30 pulses per minute AND the location in these coordinates is safe THEN send to coordinates a search and rescue team."""
    hard_rule = """IF house on fire is recognized at coordinates AND person has voice sign 'help' THEN send alert to the fire station AND send To coordinates a search and rescue team  AND send To coordinates a flying drone agent AND send To coordinates a spot robot agent."""
    
    prompt1 = header + context + easy_rule + footer
    prompt2 = header + context + medium_rule + footer
    prompt3 = header + context + hard_rule + footer

    return [prompt1, prompt2, prompt3]
  
  
  @staticmethod
  def rules():
    easy_rule = """IF Gas Is Recognized At Coordinates AND Fire Is Recognized At these Coordinates AND Agents Is Nearby to these Coordinates THEN Send Alert to these Agents AND Agents Abort Mission."""
    medium_rule = """IF a Person Is Recognized At Coordinates AND Has Heart Rate less than 30 pulses per minute AND the Location in these Coordinates Is Safe THEN Send To coordinates a Search and Rescue Team."""
    hard_rule = """IF House On Fire Is Recognized At Coordinates AND Person has Voice Sign 'help' THEN Send Alert to Fire Station AND Send To coordinates a Search and Rescue Team  AND Send To coordinates a Flying Drone Agent AND Send To coordinates a Spot Robot Agent."""
    
    prompt1 = easy_rule
    prompt2 = medium_rule
    prompt3 = hard_rule

    return [prompt1, prompt2, prompt3]
