import pytest
import os
from pyult import recording

TEST_ROOT = os.path.dirname(__file__)

def test_read_ult ():
    obj = recording.Recording()
    path = os.path.join(TEST_ROOT, 'resources/sample.ult')
    obj.read_ult(path)
    assert hasattr(obj, 'vector')

def test_read_ustxt ():
    obj = recording.Recording()
    path = os.path.join(TEST_ROOT, 'resources/sampleUS.txt')
    obj.read_ustxt(path)
    assert len(obj.__dict__)==10

def test_read_txt ():
    obj = recording.Recording()
    path = os.path.join(TEST_ROOT, 'resources/sample.txt')
    obj.read_txt(path)
    assert len(obj.__dict__)==3

def test_vec_to_imgs ():
    obj = recording.Recording()
    ultpath = os.path.join(TEST_ROOT, 'resources/sample.ult')
    uspath = os.path.join(TEST_ROOT, 'resources/sampleUS.txt')
    obj.read_ult(ultpath)
    obj.read_ustxt(uspath)
    obj.vec_to_imgs()
    assert obj.imgs.shape == (333, 842, 64)

def test_crop ():
    obj = recording.Recording()
    ultpath = os.path.join(TEST_ROOT, 'resources/sample.ult')
    uspath = os.path.join(TEST_ROOT, 'resources/sampleUS.txt')
    obj.read_ult(ultpath)
    obj.read_ustxt(uspath)
    obj.vec_to_imgs()
    obj.crop((10,50,100,700))
    assert obj.imgs.shape == (333, 601, 41)

def test_add_crop_line ():
    obj = recording.Recording()
    ultpath = os.path.join(TEST_ROOT, 'resources/sample.ult')
    uspath = os.path.join(TEST_ROOT, 'resources/sampleUS.txt')
    obj.read_ult(ultpath)
    obj.read_ustxt(uspath)
    obj.vec_to_imgs()
    obj.add_crop_line((10,50,100,700))
    cond1 = obj.imgs.shape == (333, 842, 64, 3)
    cond2 = len(set([ tuple(obj.imgs[0,100,i,:]) for i in range(obj.imgs.shape[2]) ]))==1
    assert all([cond1, cond2])

    

