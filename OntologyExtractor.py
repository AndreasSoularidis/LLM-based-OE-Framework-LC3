import re
import os

class OntologyExtractor:
  version = 1

  def __init__(self, experiment, filename="Ontology"):
    self.experiment = experiment
    self.path = os.path.join(os.getcwd(), "Experiments", self.experiment)
    self.filename = filename
    self.create_folders()

  def create_folders(self):
    os.makedirs(self.path, exist_ok=True)

  def export_to_ttl(self, text):
    pattern = re.compile(r'`(?:turtle|ttl)(.*?)`', re.DOTALL)
    match = pattern.search(text)
    
    if match:
      turtle_content = match.group(1).strip()
      filename = f"{self.filename}-v{OntologyExtractor.version}.ttl"
      full_path = os.path.join(self.path, filename)
      with open(full_path, "w") as file:
        file.write(turtle_content)
      print(f"SYSTEM: A new ontology version has been created. Current version: {OntologyExtractor.version}")
      OntologyExtractor.version += 1
      return turtle_content
    
  def export_last_to_ttl(self, text, version=1):
    if text is None:
      return

    # Use a regular expression to find all text inside ```turtle``` blocks
    matches = re.findall(r'`(?:turtle|ttl)(.*?)`', text, re.DOTALL)

    if not matches:
      print("SYSTEM: No turtle blocks found.")
      return None

    # Get the last match
    final_ontology = matches[-1].strip()

    filename = f"{self.filename}-v{version}.ttl"
    full_path = os.path.join(self.path, filename)
    
    with open(full_path, "w") as file:
      file.write(final_ontology)
    
    print(f"SYSTEM: A new ontology version has been created. Current version: {version}")
    return final_ontology
