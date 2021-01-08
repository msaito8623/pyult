import pyult.image as uimg
from pathlib import Path
import numpy as np

TEST_ROOT = Path(__file__).parent

def test_parallelize (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    obj.filter_imgs(frame=[20,40])
    aaa = uimg.parallelize(obj.imgs, uimg.ymax)
    assert type(aaa) is list
    assert len(aaa)==2
    assert type(aaa[0]) is dict
    assert list(aaa[0].keys())==['pos','img']
    assert type(aaa[0]['pos']) is np.int64
    assert type(aaa[0]['img']) is np.ndarray

