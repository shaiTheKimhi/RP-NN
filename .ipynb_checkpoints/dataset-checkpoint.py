import torch
import os
import math
class dataset(): #TODO: decide on inheritance
    def __init__(self, files_to_ignore=["dataset.py", "description.txt","documentation.txt", "__pycache__"], labels = "profile.txt", classification = 0, dir = "../datasets/Hydraulic"):
        self.class_num = classification
        self.dir = dir
        files = os.listdir(dir)
        files.remove(labels)
        files =  [f for f in files if f not in files_to_ignore]
        D = []
        L = []
        for f in files:
            try:
                with open(dir + "\\" + f, "r") as source:
                    print(f"loading file: {f}")
                    content = source.readlines()
                    content = [line.split("\t") for line in content]
                    content = torch.tensor([[float(c) for c in line] for line in content], dtype=torch.float32)
                    D.append(content)
            except:
                continue
        with open(dir + "\\" + labels, "r") as source:
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
    
    def split(self, ratio:int = 0.5):
        n = self.data[0].shape[0] * ratio
        train = subdataset(self.data[0][:int(math.floor(n))], self.data[1][:int(math.floor(n))], self.class_num)
        test =  subdataset(self.data[0][int(math.floor(n)) + 1 : ], self.data[1][int(math.floor(n)) + 1 : ], self.class_num)
        return train,test
    
    def to_type(self, data_type): #changes label data types
        self.data = self.data[0], torch.tensor(self.data[1], dtype=data_type) 
    
    def fit_classification(self):
        #self.to_type(torch.int64)
        values = []
        for i in range(2205):
            val = self.data[1][i][self.class_num]
            if val not in values:
                values.append(int(val))
            #self.data[1][i][self.class_num] = torch.tensor(values.index(val), dtype=torch.int64)
        print(values)
        for i in range(self.data[1].shape[0]):
            self.data[1][i][self.class_num] = torch.tensor(values.index(int(self.data[1][i][self.class_num])), dtype=torch.int64)
    
    
class subdataset(dataset):
    def __init__(self, data, labels, class_num):
        self.data = data, labels
        self.class_num = class_num

    
if "main" in __name__:
    d = dataset("../datasets/Hydraulic")
    a,b = d[0]
    print(a.shape)
    print(b)
