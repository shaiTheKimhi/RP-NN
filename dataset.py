import torch
import os
class dataset(): #TODO: decide on inheritance
    def __init__(self, files_to_ignore=["dataset.py", "description.txt","documentation.txt", "__pycache__"], labels = "profile.txt", classification = 0, dir = "../datasets/Hydraulic"):
        self.class_num = classification
        files = os.listdir(dir)
        files.remove(labels)
        files =  [f for f in files if f not in files_to_ignore]
        D = []
        L = []
        for f in files:
            try:
                with open(f, "r") as source:
                    print(f"loading file: {f}")
                    content = source.readlines()
                    content = [line.split("\t") for line in content]
                    content = torch.tensor([[float(c) for c in line] for line in content], dtype=torch.float32)
                    D.append(content)
            except:
                continue
        with open(labels, "r") as source:
                content = source.readlines()
                content = [line.split("\t") for line in content]
                content = torch.tensor([[float(c) for c in line] for line in content], dtype=torch.float32)
                L = content
        self.data = torch.cat(tuple(D), dim=1), L #NOTE: Labels (L) has 5 different classification tasks, parameter classification would determine which one to return
        
        #NOTE: needs to rewrite the __getitem_() function

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx][self.class_num]
    
    def __len__(self):
        return self.data[0].shape[0]

    def set_classification_task(classification: int = 0):
        self.class_num = classification

if "main" in __name__:
    d = dataset("../datasets/Hydraulic")
    a,b = d[0]
    print(a.shape)
    print(b)
