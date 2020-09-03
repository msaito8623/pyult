import numpy as np
import pytest
import pyult.pyult as pyult

rdir = './tests/resources'

@pytest.fixture
def ult():
    return pyult.UltPicture()

@pytest.fixture
def gen_img():
    def _img(dim):
        if dim==2:
            img = np.arange(12).reshape(4,3)
        elif dim==3:
            img = np.arange(24).reshape(2,4,3)
        else:
            raise ValueError('dim should be 2 or 3')
        return img
    return _img

@pytest.fixture
def load_img():
    def _img(grayscale):
        ult = pyult.UltPicture()
        img = ult.read_img(path='{}/ultimg_raw.png'.format(rdir), grayscale=grayscale)
        return img 
    return _img

def test_initialize (ult):
    attrs = ['paths', 'prompt', 'date', 'participant', 'numvectors', 'pixpervector', 'framesize', 'zerooffset', 'bitsperpixel', 'angle', 'kind', 'pixelspermm', 'framespersec', 'timeinsecsoffirstframe', 'vector', 'img', 'df']
    res = [ hasattr(ult, i) for i in attrs ]
    assert all(res)

def test_find_dir_recF (ult):
    paths = ult.find(rdir, '/', recursive=False)
    assert ult.name(paths[0]) == 'dummy_dir_outside'

def test_find_dir_recT (ult):
    paths = ult.find(rdir, '/', recursive=True)
    paths = [ ult.name(i) for i in paths ]
    tgts = ['dummy_dir_outside', 'dummy_dir_inside']
    res = [ i==j for i,j in zip(paths,tgts) ]
    assert all(res)

def test_find_all_recF (ult):
    paths = ult.find(rdir+'/dummy_dir_outside', '*', recursive=False)
    paths = [ ult.name(i) for i in paths ]
    tgts = ['dummy_dir_inside', 'file1.txt', 'file2.ult']
    res = [ i==j for i,j in zip(paths,tgts) ]
    assert all(res)

def test_find_all_recT (ult):
    paths = ult.find(rdir+'/dummy_dir_outside', '*', recursive=True)
    paths = [ ult.name(i) for i in paths ]
    tgts = ['dummy_dir_inside', 'file3.txt', 'file4.ult', 'file5.tar.gz', 'file1.txt', 'file2.ult']
    res = [ i==j for i,j in zip(paths,tgts) ]
    assert all(res)

def test_find_files_recF (ult):
    paths = ult.find(rdir+'/dummy_dir_outside', '.ult', recursive=False)
    paths = [ ult.name(i) for i in paths ]
    tgts = ['file2.ult']
    res = [ i==j for i,j in zip(paths,tgts) ]
    assert all(res)

def test_find_files_recT (ult):
    paths = ult.find(rdir+'/dummy_dir_outside', '.ult', recursive=True)
    paths = [ ult.name(i) for i in paths ]
    tgts = ['file4.ult', 'file2.ult']
    res = [ i==j for i,j in zip(paths,tgts) ]
    assert all(res)

def test_replace (ult):
    paths = ult.find(rdir+'/dummy_dir_outside', '.ult', recursive=False)
    paths = [ ult.replace('(\\.[A-z]+)$', '_modified\\1', i, regex=True) for i in paths ]
    paths = [ ult.name(i) for i in paths ]
    tgts = [ 'file2_modified.ult' ]
    res = [ i==j for i,j in zip(paths,tgts) ]
    assert all(res)

def test_name (ult):
    paths = ult.find(rdir+'/dummy_dir_outside', '.ult', recursive=False)
    paths = [ ult.name(i) for i in paths ]
    tgts = [ 'file2.ult' ]
    res = [ i==j for i,j in zip(paths,tgts) ]
    assert all(res)

def test_suffix (ult):
    paths = ult.find(rdir+'/dummy_dir_outside', '.ult', recursive=False)
    paths = [ ult.suffix(i) for i in paths ]
    tgts = [ '.ult' ]
    res = [ i==j for i,j in zip(paths,tgts) ]
    assert all(res)

def test_set_paths (ult):
    ult.set_paths(rdir)
    tsts = set( ult.paths.keys() )
    tgts = set( ('txt', 'ult', 'wav', 'textgrid', 'ustxt') )
    assert len(tst.difference(tgts)) == 0

def test_is_all_set (ult):
    ult.set_paths(rdir)
    assert ult.is_all_set(exclude_empty_list=True)

def test_resize (ult, load_img):
    img = load_img(grayscale=True)
    img = ult.resize(new_xy=(500,500), img=img)
    assert img.shape == (500,500)

def test_reduce_resolution (ult, gen_img):
    imgs = gen_img(3)
    ult.reduce_resolution(img=imgs, every_x=2, every_y=2, inplace=True)
    tgt = np.array([0,2,6,8,12,14,18,20]).reshape(2,2,2)
    assert (tgt!=ult.img).sum()==0



