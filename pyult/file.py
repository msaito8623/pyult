import numpy as np
import pandas as pd
from pathlib import Path
import re

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
def read_textgrid (path):
    encs = ['utf8', 'utf16', 'latin1']
    for i in encs:
        try:
            with open(path, 'r', encoding=i) as f:
                lines = f.readlines()
                lines = [ i.rstrip('\n') for i in lines ]
            break
        except UnicodeError:
            pass
    return lines
def mainpart (path):
    """
    Removes typical strings from file names of ultrasound files,
    which can be problematic for checking correspondence of
    ultrasound files, e.g. 'xxx_Track0' --> 'xxx'.
    """
    rmvs = ['_Track[0-9]+$', 'US$', '_corrected']
    for i in rmvs:
        path = re.sub(i, '', path)
    return path
def check_wdir (path, verbose):
    """
    Receives a path to a target directory supposedly containg
    all the necessary files for preprocessing of ultrasound images
    and checks if really all the necessary files are ready.
    """
    wdir = Path(path)
    extensions = ['.phoneswithQ', '.TextGrid', '.ult', '.wav', '.words', '[!S].txt', 'US.txt']
    paths = [ wdir.glob('*{}'.format(i)) for i in extensions ]
    paths = [ list(i) for i in paths ]
    paths = [ i for i in paths if len(i)!=0 ]
    lens = [ len(i) for i in paths ]
    if len(set(lens))!=1:
        if verbose:
            print('###')
            print('Lengths of each file type differ.')
            print('###')
        res = False
    else:
        paths = [ [ j.stem for j in i ] for i in paths ]
        paths = [ sorted(i) for i in paths ]
        paths = [ [ mainpart(j) for j in i ] for i in paths ]
        res = True
        for i in range(len(paths[0])):
            ith_item_each_pathtype = [ j[i] for j in paths ]
            if len(set(ith_item_each_pathtype))!=1:
                if verbose:
                    print('###')
                    print('Lengths of each file type are fine. But file stems do not correspond.')
                    print('###')
                res = False
                break
    return res
def unique_target_stems (path):
    """
    Generates unique file stems from a path to a target directory.
    Note that this function does not check correspondence of files.
    """
    wdir = Path(path)
    extensions = ['.phoneswithQ', '.TextGrid', '.ult', '.wav', '.words', '[!S].txt', 'US.txt']
    paths = [ wdir.glob('*{}'.format(i)) for i in extensions ]
    paths = [ j for i in paths for j in i ]
    paths = [ mainpart(i.stem) for i in paths ]
    paths = sorted(list(set(paths)))
    return paths
def find_target_file (wdir, stem, extension, no_hit_ok=True, recursive=False):
    """
    Find a target file path from a path to the working directory, the stem of the file name, and the extension of the target file.
    """
    if recursive:
        paths = list(Path(wdir).glob('**/{}*'.format(stem)))
    else:
        paths = list(Path(wdir).glob('{}*'.format(stem)))
    path = [ i for i in paths if re.search(extension, str(i)) ]
    if len(path)==0:
        if no_hit_ok:
            ret = ''
        else:
            raise ValueError('No file is matched.')
    elif len(path)>1:
        raise ValueError('More than one file is matched.')
    else:
        ret = str(path[0])
    return ret
