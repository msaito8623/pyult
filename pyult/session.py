from pyult.recording import Recording
from pathlib import Path
import re
import pandas as pd
import pyult.file as ufile

class Session:
    def __init__ (self, wdir):
        self.wdir = wdir
        paths = Path(wdir).glob('*')
        paths = [ str(i) for i in paths if i.is_file() ]
        self.paths = dict()
        self.paths['phn'] = sorted([ i for i in paths if re.search('\.phoneswithQ$', i) ])
        self.paths['tgd'] = sorted([ i for i in paths if re.search('\\.TextGrid$', i) ])
        self.paths['wav'] = sorted([ i for i in paths if re.search('\\.wav$', i) ])
        self.paths['txt'] = sorted([ i for i in paths if re.search('[^S]\\.txt$', i) ])
        self.paths['ult'] = sorted([ i for i in paths if re.search('\\.ult$', i) ])
        self.paths['ust'] = sorted([ i for i in paths if re.search('US\\.txt$', i) ])
        self.paths['wrd'] = sorted([ i for i in paths if re.search('\\.words$', i) ])
        self.basenames = [ Path(i).stem for i in self.paths['ult'] ]
    def check (self):
        if len(set([ len(i) for i in self.paths.values() ])) != 1:
            ok = False
        else:
            paths = pd.DataFrame(self.paths)
            if len(set(paths.apply(lambda x: len(set([ ufile.mainpart(Path(i).stem) for i in x ])), axis=1))) != 1:
                ok = False
            else:
                ok = True
        return ok
    def inspect (self):
        print('This method is yet to be implemented...')
        return None
    def load_recording (self, basename_or_index):
        ind = [ i for i,j in enumerate(self.basenames) if j==basename_or_index ]
        assert len(ind)==1
        ind = ind[0]
        rec = Recording()
        rec.read_ult(self.paths['ult'][ind])
        rec.read_ustxt(self.paths['ust'][ind])
        rec.read_txt(self.paths['txt'][ind])
        rec.read_phones(self.paths['phn'][ind])
        rec.read_words(self.paths['wrd'][ind])
        rec.read_textgrid(self.paths['tgd'][ind])
        return rec


