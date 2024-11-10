import os

class ChatExtractor:

  def __init__(self, experiment, filename="Chat"):
    self.experiment = experiment
    self.path = os.path.join(os.getcwd(), "Experiments", self.experiment)
    self.filename = filename
    self.create_folders()

  def create_folders(self):
    os.makedirs(self.path, exist_ok=True)
    
  def discussion_extractor(self, text):
    if text == None:
      return
    filename = f"{self.filename}.txt"
    full_path = os.path.join(self.path, filename)
    with open(full_path, "w") as file:
      file.write(text)
    print(f"SYSTEM: A file that contains all the chat discussion has been created.")
    return