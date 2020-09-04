import cv2
import numpy as np
import os
import pandas as pd
import pytest
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

def test_read_phones ():
    obj = recording.Recording()
    path = os.path.join(TEST_ROOT, 'resources/sample.phoneswithQ')
    obj.read_phones(path)
    cond1 = len(obj.phones)==13
    cond2 = obj.phones.loc[10,'end']==1.51
    assert all([cond1, cond2])

def test_read_words ():
    obj = recording.Recording()
    path = os.path.join(TEST_ROOT, 'resources/sample.words')
    obj.read_words(path)
    cond1 = len(obj.words)==7
    cond2 = obj.words.loc[3,'end']==0.93
    assert all([cond1, cond2])

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

def test_fit_spline_without_fitted_values ():
    obj = recording.Recording()
    ultpath = os.path.join(TEST_ROOT, 'resources/sample.ult')
    uspath = os.path.join(TEST_ROOT, 'resources/sampleUS.txt')
    obj.read_ult(ultpath)
    obj.read_ustxt(uspath)
    obj.vec_to_imgs()
    obj.imgs = obj.imgs[:2]
    obj.fit_spline(set_fitted_values=False)
    cond1 = obj.imgs.shape == (2, 842, 64, 3)
    figpath = os.path.join(TEST_ROOT, 'resources/sample_spline.png')
    img = cv2.imread(figpath, 1)
    cond2 = (obj.imgs[1] == img).all()
    assert all([cond1, cond2])

def test_fit_spline_with_fitted_values ():
    obj = recording.Recording()
    ultpath = os.path.join(TEST_ROOT, 'resources/sample.ult')
    uspath = os.path.join(TEST_ROOT, 'resources/sampleUS.txt')
    obj.read_ult(ultpath)
    obj.read_ustxt(uspath)
    obj.vec_to_imgs()
    obj.imgs = obj.imgs[:2]
    obj.fit_spline(set_fitted_values=True)
    assert hasattr(obj, 'fitted_values')

def test_imgs_to_df ():
    obj = recording.Recording()
    ultpath = os.path.join(TEST_ROOT, 'resources/sample.ult')
    uspath = os.path.join(TEST_ROOT, 'resources/sampleUS.txt')
    obj.read_ult(ultpath)
    obj.read_ustxt(uspath)
    obj.vec_to_imgs()
    obj.imgs_to_df()
    pos = np.linspace(0, len(obj.df)-1, 10).round().astype(int)
    obj.df = obj.df.iloc[pos,:].reset_index(drop=True).round(10)
    gzpath = os.path.join(TEST_ROOT, 'resources/sample.gz')
    samplegz = pd.read_csv(gzpath, sep='\t', header=0)
    samplegz = samplegz.round(10)
    assert (obj.df==samplegz).all().all()

def test_integrate_segments ():
    obj = recording.Recording()
    ultpath = os.path.join(TEST_ROOT, 'resources/sample.ult')
    uspath = os.path.join(TEST_ROOT, 'resources/sampleUS.txt')
    phonespath = os.path.join(TEST_ROOT, 'resources/sample.phoneswithQ')
    wordspath = os.path.join(TEST_ROOT, 'resources/sample.words')
    obj.read_ult(ultpath)
    obj.read_ustxt(uspath)
    obj.read_phones(phonespath)
    obj.read_words(wordspath)
    obj.vec_to_imgs()
    obj.imgs_to_df()
    obj.integrate_segments()
    pos = np.linspace(0, len(obj.df)-1, 10).round().astype(int)
    obj.df = obj.df.iloc[pos,:].reset_index(drop=True).round(10)
    gzpath = os.path.join(TEST_ROOT, 'resources/sample2.gz')
    samplegz = pd.read_csv(gzpath, sep='\t', header=0)
    samplegz = samplegz.round(10)
    assert (obj.df==samplegz).all().all()

def test_integrate_splines ():
    obj = recording.Recording()
    ultpath = os.path.join(TEST_ROOT, 'resources/sample.ult')
    uspath = os.path.join(TEST_ROOT, 'resources/sampleUS.txt')
    obj.read_ult(ultpath)
    obj.read_ustxt(uspath)
    obj.vec_to_imgs()
    obj.imgs = obj.imgs[:2]
    obj.imgs_to_df()
    obj.integrate_splines()
    cond1 = 'y_spline' in obj.df.columns
    cond2 = len(set(obj.df.y_spline.dropna())) == 68
    assert all([cond1, cond2])





    

