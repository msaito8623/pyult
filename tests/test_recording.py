import cv2
import numpy as np
import os
import pandas as pd
import pytest
from pyult import recording
import shutil
from pathlib import Path
import pyult.image as uimg

TEST_ROOT = Path(__file__).parent

def test_read_ult (rec_obj):
    obj = rec_obj(par=False)
    path = str(TEST_ROOT / 'resources/sample_recording/sample_01.ult')
    obj.read_ult(path)
    assert hasattr(obj, 'vector')

def test_read_ustxt (rec_obj):
    obj = rec_obj(par=False)
    path = str(TEST_ROOT / 'resources/sample_recording/sample_01US.txt')
    obj.read_ustxt(path)
    assert len(obj.__dict__)==10

def test_read_txt (rec_obj):
    obj = rec_obj(par=False)
    path = str(TEST_ROOT / 'resources/sample_recording/sample_01.txt')
    obj.read_txt(path)
    assert len(obj.__dict__)==3

def test_read_phones (rec_obj):
    obj = rec_obj(par=False)
    path = str(TEST_ROOT / 'resources/sample_recording/sample_01.phoneswithQ')
    obj.read_phones(path)
    cond1 = len(obj.phones)==13
    cond2 = obj.phones.loc[10,'end']==1.51
    assert all([cond1, cond2])

def test_read_words (rec_obj):
    obj = rec_obj(par=False)
    path = str(TEST_ROOT / 'resources/sample_recording/sample_01.words')
    obj.read_words(path)
    cond1 = len(obj.words)==7
    cond2 = obj.words.loc[3,'end']==0.93
    assert all([cond1, cond2])

def test_read_textgrid (rec_obj):
    obj = rec_obj(par=False)
    path = str(TEST_ROOT / 'resources/sample_recording/sample_01.TextGrid')
    obj.read_textgrid(path)
    cond1 = isinstance(obj.textgrid, list)
    cond2 = len(obj.textgrid)>0
    assert all([cond1, cond2])

def test_vec_to_imgs (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    assert obj.imgs.shape == (333, 842, 64)

def test_crop (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    obj.crop((10,50,100,700))
    assert obj.imgs.shape == (333, 601, 41)

def test_add_crop_line (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    obj.add_crop_line((10,50,100,700))
    cond1 = obj.imgs.shape == (333, 842, 64, 3)
    cond2 = len(set([ tuple(obj.imgs[0,100,i,:]) for i in range(obj.imgs.shape[2]) ]))==1
    assert all([cond1, cond2])

def test_flip_x (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    tesvec = obj.imgs[0,0,:]
    tesvec = tesvec[::-1]
    obj.flip('x')
    assert all(obj.imgs[0,0,:] == tesvec)

def test_flip_y (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    tesvec = obj.imgs[0,:,0]
    tesvec = tesvec[::-1]
    obj.flip('y')
    assert all(obj.imgs[0,:,0] == tesvec)
    
def test_flip_xy (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    tesvecx = obj.imgs[0,0,:]
    tesvecx = tesvecx[::-1]
    tesvecy = obj.imgs[0,:,0]
    tesvecy = tesvecy[::-1]
    obj.flip('xy')
    cond1 = all(obj.imgs[0,-1,:] == tesvecx)
    cond2 = all(obj.imgs[0,:,-1] == tesvecy)
    assert all([cond1, cond2])

def test_reduce_y (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    tesvec = obj.imgs[0,:,0]
    tesvec = tesvec[::3]
    obj.reduce_y(3)
    cond1 = obj.imgs.shape == (333, 281, 64)
    cond2 = all(obj.imgs[0,:,0] == tesvec)
    assert all([cond1, cond2])

def test_fit_spline_images (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    obj.imgs = obj.imgs[:2]
    obj.fit_spline()
    cond1 = obj.splimgs.shape == (2, 842, 64, 3)
    figpath = str(TEST_ROOT / 'resources/sample_spline.png')
    img = cv2.imread(figpath, 1)
    cond2 = np.equal(obj.splimgs[1], img).all()
    assert all([cond1, cond2])

def test_fit_spline_values (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    obj.imgs = obj.imgs[:2]
    obj.fit_spline(fitted_images=False, fitted_values=True)
    assert hasattr(obj, 'fitted_values')

def test_imgs_to_df (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    obj.imgs_to_df()
    pos = np.linspace(0, len(obj.df)-1, 10).round().astype(int)
    obj.df = obj.df.iloc[pos,:].reset_index(drop=True).round(10)
    gzpath = os.path.join(TEST_ROOT, 'resources/sample.gz')
    samplegz = pd.read_csv(gzpath, sep='\t', header=0)
    samplegz = samplegz.round(10)
    assert (obj.df==samplegz).all().all()

def test_integrate_segments (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    obj.imgs_to_df()
    obj.integrate_segments()
    pos = np.linspace(0, len(obj.df)-1, 10).round().astype(int)
    obj.df = obj.df.iloc[pos,:].reset_index(drop=True).round(10)
    gzpath = os.path.join(TEST_ROOT, 'resources/sample2.gz')
    samplegz = pd.read_csv(gzpath, sep='\t', header=0)
    samplegz = samplegz.round(10)
    assert (obj.df==samplegz).all().all()

def test_rmv_noise (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    obj.imgs_to_df()
    obj.integrate_segments()
    obj.rmv_noise()
    assert len(set(obj.df.word)) == 2

def test_integrate_splines (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    obj.imgs = obj.imgs[:2]
    obj.imgs_to_df()
    obj.integrate_splines()
    cond1 = 'y_spline' in obj.df.columns
    cond2 = len(set(obj.df.y_spline.dropna())) == 65
    assert all([cond1, cond2])

def test_textgrid_to_alignfiles (rec_obj):
    obj = rec_obj(par=True)
    obj.textgrid_to_alignfiles()
    cond1 = hasattr(obj, 'phones')
    cond2 = hasattr(obj, 'words')
    assert all([cond1, cond2])

def test_textgrid_to_df (rec_obj):
    obj = rec_obj(par=True)
    obj.textgrid_to_df()
    cond1 = hasattr(obj, 'textgrid_df')
    cond2 = len(obj.textgrid_df)>0
    clms = ['end','segments','words']
    cond3 = all([ i==j for i,j in zip(clms,obj.textgrid_df.columns) ])
    assert all([cond1, cond2, cond3])

def test_square_imgs (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    obj.square_imgs()
    cond1 = hasattr(obj, 'squares')
    cond2 = all([ len(set(i.shape))==1 for i in obj.squares ])
    assert all([cond1, cond2])

def test_to_fan (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    obj.filter_imgs(frame=[20,40,60])
    obj.to_fan(magnify=3)
    paths = [ 'resources/sample_fan_{:03d}.png'.format(i) for i in range(3) ]
    paths = [ str(TEST_ROOT / Path(i)) for i in paths ]
    fans = [ uimg.read_img(i) for i in paths ]
    check = all([ np.equal(i,j).all() for i,j in zip(obj.fans, fans) ])
    assert check

def test_to_fan_general_parameters (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    obj.filter_imgs(frame=[20,40,60])
    obj.to_fan(general_parameters=True)
    paths = [ 'resources/sample_fan_genpar_{:03d}.png'.format(i) for i in range(3) ]
    paths = [ str(TEST_ROOT / Path(i)) for i in paths ]
    fans = [ uimg.read_img(i) for i in paths ]
    check = all([ np.equal(i,j).all() for i,j in zip(obj.fans, fans) ])
    assert check

def test_read_img (rec_obj):
    obj = rec_obj(par=False)
    gry_path = str(TEST_ROOT / 'resources/sample_fan_000.png')
    obj.read_img(gry_path, True)
    gry2 = cv2.imread(gry_path, 0)
    gry_check = np.equal(obj.img, gry2).all()
    rgb_path = str(TEST_ROOT / 'resources/sample_spline.png')
    obj.read_img(rgb_path, False)
    rgb2 = cv2.imread(rgb_path, 1)
    rgb_check = np.equal(obj.img, rgb2).all()
    assert gry_check and rgb_check

def test_save_img (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    temp_dir = TEST_ROOT / 'resources/temp'
    os.makedirs(str(temp_dir), exist_ok=True)
    obj.save_img(str(temp_dir / 'temp.png'))
    paths = list(temp_dir.glob('*.png'))
    ok = len(paths)==len(obj.imgs)
    shutil.rmtree(str(temp_dir))
    assert ok

def test_filter_imgs (rec_obj):
    obj = rec_obj(par=True)
    obj.vec_to_imgs()
    aaa = obj.filter_imgs(frame=1, inplace=False)
    bbb = obj.filter_imgs(time=0.0245, inplace=False)
    obj.filter_imgs(time=0.0245)
    tst1 = np.equal(aaa,bbb).all()
    tst2 = np.equal(aaa,obj.imgs).all()
    assert tst1 and tst2

def test_ymax (rec_obj):
    obj = rec_obj(par=False)
    img = str(TEST_ROOT / 'resources/sample_fan_000.png')
    obj.read_img(img, True)
    obj.ymax(return_img=True)
    assert type(obj.ymaxpos) is np.int64
    assert obj.ymaxpos > 0
    assert type(obj.ymaximgs) is np.ndarray
    assert len(obj.ymaximgs.shape) == 3

    

