import pytest
from pathlib import Path
from pyult.recording import Recording
import pyult.image as uimg

TEST_ROOT = Path(__file__).parent

@pytest.fixture
def rec_obj ():
    def _rec_obj (par=True):
        rec = Recording()
        if par:
            wdr = TEST_ROOT / Path('resources/sample_recording')
            fnm = {'ult':'sample_01.ult',
                   'ust':'sample_01US.txt',
                   'txt':'sample_01.txt',
                   'phn':'sample_01.phoneswithQ',
                   'wrd':'sample_01.words',
                   'tgd':'sample_01.TextGrid'}
            fnm = { i:wdr/Path(j) for i,j in fnm.items() }
            rec.read_ult(fnm['ult'])
            rec.read_ustxt(fnm['ust'])
            rec.read_txt(fnm['txt'])
            rec.read_phones(fnm['phn'])
            rec.read_words(fnm['wrd'])
            rec.read_textgrid(fnm['tgd'])
        return rec
    return _rec_obj
