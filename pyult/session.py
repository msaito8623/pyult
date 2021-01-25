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
    def check (self, ignore_empty_file_type=True):
        if ignore_empty_file_type:
            paths = { i:j for i,j in self.paths.items() if len(j)!=0 }
        else:
            paths = self.paths
        if len(set([ len(i) for i in paths.values() ])) != 1:
            ok = False
        else:
            paths = pd.DataFrame(paths)
            if len(set(paths.apply(lambda x: len(set([ ufile.mainpart(Path(i).stem) for i in x ])), axis=1))) != 1:
                ok = False
            else:
                ok = True
        return ok
    def inspect (self, ignore_empty_file_type=True):
        ok = self.check(ignore_empty_file_type)
        if ok:
            print('The working directory is fine and ready for preprocessing.')
        else:
            if ignore_empty_file_type:
                paths = { i:j for i,j in self.paths.items() if len(j)!=0 }
            else:
                paths = self.paths
            if len(set([ len(i) for i in paths.values() ])) != 1:
                print('Numbers of files in each file type do not match.')
                nms = {'wav':'        .wav', 'txt':'        .txt',
                        'ult':'        .ult', 'ust':'      US.txt',
                        'wrd':'      .words', 'phn':'.phoneswithQ',
                        'tgd':'   .TextGrid'}
                for i,j in paths.items():
                    print('{} --> {: 3d} files'.format(nms[i],len(j)))
            else:
                paths = pd.DataFrame(paths)
                if len(set(paths.apply(lambda x: len(set([ ufile.mainpart(Path(i).stem) for i in x ])), axis=1))) != 1:
                    aaa = paths.apply(lambda x: len(set([ ufile.mainpart(Path(i).stem) for i in x ])), axis=1)
                    err_rows = paths.loc[aaa.loc[aaa!=1].index,]
                    err_basenames = err_rows.apply(lambda x: set([ ufile.mainpart(Path(i).stem) for i in x ]), axis=1)
                    err_bn_set = set()
                    for i in err_basenames:
                        err_bn_set = err_bn_set.union(i)
                    print('Filenames do not match. Following filenames may be a problem.')
                    for i in sorted(list(err_bn_set)):
                        print(i)
        return None
    def load_recording (self, basename_or_index):
        if isinstance(basename_or_index, int):
            ind = [basename_or_index]
        else:
            ind = [ i for i,j in enumerate(self.basenames) if j==basename_or_index ]
        assert len(ind)==1
        ind = ind[0]
        rec = Recording()
        ftypes = ['ult', 'ust', 'txt', 'tgd', 'phn', 'wrd']
        for i in ftypes:
            try:
                rec.read(self.paths[i][ind])
            except IndexError:
                pass
        return rec


