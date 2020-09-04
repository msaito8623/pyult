import pytest
import os
import cv2
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

def test_flip_x ():
    obj = recording.Recording()
    ultpath = os.path.join(TEST_ROOT, 'resources/sample.ult')
    uspath = os.path.join(TEST_ROOT, 'resources/sampleUS.txt')
    obj.read_ult(ultpath)
    obj.read_ustxt(uspath)
    obj.vec_to_imgs()
    tesvec = obj.imgs[0,0,:]
    tesvec = tesvec[::-1]
    obj.flip('x')
    assert all(obj.imgs[0,0,:] == tesvec)

def test_flip_y ():
    obj = recording.Recording()
    ultpath = os.path.join(TEST_ROOT, 'resources/sample.ult')
    uspath = os.path.join(TEST_ROOT, 'resources/sampleUS.txt')
    obj.read_ult(ultpath)
    obj.read_ustxt(uspath)
    obj.vec_to_imgs()
    tesvec = obj.imgs[0,:,0]
    tesvec = tesvec[::-1]
    obj.flip('y')
    assert all(obj.imgs[0,:,0] == tesvec)
    
def test_flip_xy ():
    obj = recording.Recording()
    ultpath = os.path.join(TEST_ROOT, 'resources/sample.ult')
    uspath = os.path.join(TEST_ROOT, 'resources/sampleUS.txt')
    obj.read_ult(ultpath)
    obj.read_ustxt(uspath)
    obj.vec_to_imgs()
    tesvecx = obj.imgs[0,0,:]
    tesvecx = tesvecx[::-1]
    tesvecy = obj.imgs[0,:,0]
    tesvecy = tesvecy[::-1]
    obj.flip('xy')
    cond1 = all(obj.imgs[0,-1,:] == tesvecx)
    cond2 = all(obj.imgs[0,:,-1] == tesvecy)
    assert all([cond1, cond2])

def test_reduce_y ():
    obj = recording.Recording()
    ultpath = os.path.join(TEST_ROOT, 'resources/sample.ult')
    uspath = os.path.join(TEST_ROOT, 'resources/sampleUS.txt')
    obj.read_ult(ultpath)
    obj.read_ustxt(uspath)
    obj.vec_to_imgs()
    tesvec = obj.imgs[0,:,0]
    tesvec = tesvec[::3]
    obj.reduce_y(3)
    cond1 = obj.imgs.shape == (333, 281, 64)
    cond2 = all(obj.imgs[0,:,0] == tesvec)
    assert all([cond1, cond2])

def test_fit_spline ():
    obj = recording.Recording()
    ultpath = os.path.join(TEST_ROOT, 'resources/sample.ult')
    uspath = os.path.join(TEST_ROOT, 'resources/sampleUS.txt')
    obj.read_ult(ultpath)
    obj.read_ustxt(uspath)
    obj.vec_to_imgs()
    obj.imgs = obj.imgs[:2]
    obj.fit_spline()
    cond1 = obj.imgs.shape == (3, 842, 64, 3)
    figpath = os.path.join(TEST_ROOT, 'resources/sample_spline.png')
    img = cv2.imread(figpath, 1)
    assert (obj.imgs[1] == img).all()
    





    

