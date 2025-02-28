import torch
import os


def  save(self, path):

    data = {}

    for name in vars(self):
        if name not in {"live"}:
            data[name] = getattr(self, name)


    torch.save(data, path + "~")
    os.rename(path + "~", path)

def load(self, path):
    data = torch.load(path)
    #print(dir(self))
    for key, value in data.items():
        setattr(self, key, value)
        #print(dir(self))
        # if hasattr(self, key):
        #     setattr(self, key, value)
        # else:
        #     print("No '{}' in the class".format(key))
    #print(dir(self))