import numpy as np
import pandas as pd

def read_ult (path):
    return np.fromfile(open(path, "rb"), dtype=np.uint8)
def read_ustxt (path):
    with open(path, "r") as f:
        content = f.readlines()
    content = [ i.rstrip('\n') for i in content ]
    content = [ i.split('=') for i in content ]
    content = { i[0]:i[1] for i in content }
    content = { i: int(j) if j.isdigit() else float(j) for i,j in content.items() }
    content['FrameSize'] = content['NumVectors'] * content['PixPerVector']
    return content
def read_txt (path):
    with open(path, "r", encoding='latin1') as f:
        content = f.readlines()
    content = [ i.rstrip('\n') for i in content ]
    content = [ i.rstrip(',') for i in content ]
    kys = ['prompt', 'date', 'participant']
    content = { i:j for i,j in zip(kys, content) }
    return content
def read_phones (path):
    colns = ['end', 'segment']
    dat = pd.read_csv(path, sep=' ', header=None, skiprows=[0], usecols=[0,2], names=colns)
    return dat
def read_words (path):
    colns = ['end','word']
    dat = pd.read_csv(path, sep=' ', header=None, skiprows=[0], usecols=[0,2], names=colns)
    return dat


