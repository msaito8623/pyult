import copy
import os
import pandas as pd
import pyult.file as file
import pyult.image as image 
import pyult.dataframe as dataframe
from pathlib import Path
import numpy as np

class Recording:
    def __init__ (self):
        pass
    def dict_to_attr ( self, dct ):
        for i,j in dct.items():
            setattr(self, i, j)
        return None
    def read (self, path):
        ftype = file.which_filetype(path)
        tofunc = {'ult':self.read_ult, 'ust':self.read_ustxt, 'txt':self.read_txt, 'tgd':self.read_textgrid, 'phn':self.read_phones, 'wrd':self.read_words}
        tofunc[ftype](path)
        return None
    def read_ult (self, path):
        self.vector = file.read_ult(path)
        return None
    def read_ustxt (self, path):
        dct = file.read_ustxt(path)
        self.dict_to_attr(dct=dct)
        return None
    def read_txt (self, path):
        dct = file.read_txt(path)
        self.dict_to_attr(dct=dct)
        return None
    def read_phones (self, path):
        try:
            self.phones = file.read_phones(path)
        except FileNotFoundError:
            if not hasattr(self, 'phones'):
                self.phones = None
            else:
                pass
        return None
    def read_words (self, path):
        try:
            self.words = file.read_words(path)
        except FileNotFoundError:
            self.words = None
        return None
    def read_textgrid (self, path):
        try:
            self.textgrid = file.read_textgrid(path)
        except FileNotFoundError:
            self.textgrid = None
        return None
    def read_frames_csv (self, path, sep=',', header=None):
        try:
            with open(path, 'r') as f:
                frms = f.readlines()
            frms = [ i.strip().split(sep) for i in frms ]
            frms = { i[0]:i[1:] for i in frms }
            self.pkdframes = frms
        except FileNotFoundError:
            self.pkdframes = None
        return None
    def read_img (self, inpath, grayscale=True):
        self.img = image.read_img(inpath, grayscale)
        return None
    def save_img (self, outpath, img=None):
        if img is None:
            if hasattr(self, 'img'):
                img = self.img
            elif hasattr(self, 'imgs'):
                img = self.imgs
            else:
                raise ValueError('Image to be saved cannot be found.')
        image.save_img(outpath, img)
        return None
    def vec_to_imgs (self):
        self.imgs = image.vec_to_imgs(self.vector, self.NumVectors, self.PixPerVector)
        return None
    def filter_imgs (self, frame=None, time=None, inplace=True):
        if (not time is None) and (not hasattr(self, 'FramesPerSec')):
            raise AttributeError('Frame rate is necessary when filtering by time. Load a parameter file first with self.read_ustxt.')
        filtered_imgs = image.filter_imgs(self.imgs, frame, time, self.FramesPerSec)
        if inplace:
            self.imgs = filtered_imgs
            rtn = None
        else:
            rtn = filtered_imgs
        return rtn
    def filter_by_segments (self, segments, duplicates='last', nohit='untouched'):
        if not hasattr(self, 'textgrid_df'):
            self.textgrid_to_df()
        tdf = self.textgrid_df.copy()
        if duplicates=='first':
            pos = 0
        elif duplicates=='last':
            pos = -1
        else:
            raise ValueError('"duplicates" must be "first" or "last".')
        segpos = tdf.segments.isin(pd.Series(segments))
        if any(segpos):
            tim = tdf.loc[segpos,['start_segments','end_segments']].reset_index(drop=True).iloc[pos].to_list()
            frames = image.time_to_frame(time=tim, fps=self.FramesPerSec, len_frames = len(self.imgs))
            frames[1] = frames[1] + 1
            frames = range(*frames)
            self.filter_imgs(frame=frames)
            self.segment = tdf.segments.loc[segpos].iloc[-1]
        else:
            if nohit=='untouched':
                pass
            elif nohit=='none':
                self.imgs = None
            else:
                raise ValueError('"nohit" must be "untouched" or "none".')
        return None
    def crop (self, crop_points):
        self.imgs = image.crop(self.imgs, crop_points)
        return None
    def add_crop_line (self, crop_points):
        self.imgs = image.add_crop_line(self.imgs, crop_points)
        return None
    def flip (self, direct):
        self.imgs = image.flip(self.imgs, direct)
        return None
    def reduce_y (self, every_nth):
        self.imgs = image.reduce_y(self.imgs, every_nth)
        return None
    def fit_spline (self, fitted_images=True, fitted_values=False):
        imgs_ftvs = image.fit_spline(self.imgs, fitted_values, fitted_images)
        if fitted_images and fitted_values:
            self.splimgs = imgs_ftvs['images']
            self.fitted_values = imgs_ftvs['fitted_values']
        elif (not fitted_images) and fitted_values:
            self.fitted_values = imgs_ftvs
        elif fitted_images and (not fitted_values):
            self.splimgs = imgs_ftvs
        else:
            pass
        return None
    def imgs_to_df (self, frame_id='', inplace=True):
        if inplace:
            if len(self.imgs.shape)==3:
                self.df = dataframe.imgs_to_df(self.imgs, self.FramesPerSec)
            elif len(self.imgs.shape)==2:
                self.df = dataframe.img_to_df(self.imgs, frame_id, self.FramesPerSec)
            obj = None
        else:
            obj = copy.deepcopy(self)
            if len(obj.imgs.shape)==3:
                obj.df = dataframe.imgs_to_df(obj.imgs, obj.FramesPerSec)
            elif len(obj.imgs.shape)==2:
                obj.df = dataframe.img_to_df(obj.imgs, frame_id, obj.FramesPerSec)
        return obj
    def integrate_segments (self):
        try:
            c1 = not self.phones is None
            c2 = not self.words is None
            cond1 = all([c1,c2])
        except AttributeError:
            cond1 = False
        try:
            cond2 = not self.textgrid is None
        except AttributeError:
            cond2 = False

        if cond1:
            self.df = dataframe.integrate_segments(self.df, self.phones, self.words)
        elif cond2:
            self.textgrid_to_alignfiles()
            self.df = dataframe.integrate_segments(self.df, self.phones, self.words)
        else:
            print('WARNING: There is no .TextGrid, .phones, .phoneswithQ, or .words available. Therefore, the result dataframe does not contain segment information.')
        return None
    def rmv_noise (self):
        self.df = dataframe.rmv_noise(self.df)
        return None
    def integrate_splines (self):
        if not hasattr(self, 'fitted_values'):
            self.fit_spline(fitted_images=False, fitted_values=True)
        self.df = dataframe.integrate_splines(self.df, self.fitted_values)
        return None
    def textgrid_to_alignfiles (self):
        aligns = dataframe.textgrid_to_alignfiles(self.textgrid)
        self.words= aligns['words']
        self.phones = aligns['segments']
        return None
    def textgrid_to_df (self):
        self.textgrid_df = dataframe.textgrid_to_df(self.textgrid)
        return None
    def square_imgs (self, size=None, inplace=True, attr='squares'):
        squ = image.to_square(self.imgs, size)
        if inplace:
            setattr(self, attr, squ)
            squ = None
        return squ
    def to_fan (self, imgs=None, general_parameters=False, magnify=1, show_progress=False ):
        if imgs is None:
            imgs = self.imgs
        if general_parameters:
            self.fans = image.to_fan(imgs, magnify=magnify, show_progress=show_progress)
        else:
            self.fans = image.to_fan(imgs, self.Angle, self.ZeroOffset, self.PixelsPerMm, self.NumVectors, magnify=magnify, show_progress=show_progress)
        return None
    def write_video (self, audiopath, outpath='./video.avi'):
        vtemp = Path(outpath)
        vtemp = str(vtemp.parent) + '/' + vtemp.stem + '_temp' + vtemp.suffix
        atemp = Path(audiopath)
        atemp = str(atemp.parent) + '/' + atemp.stem + '_temp' + atemp.suffix
        image._temp_video(self.imgs, self.FramesPerSec, vtemp)
        image._temp_audio(audiopath, self.FramesPerSec, atemp, self.TimeInSecsOfFirstFrame)
        image.sync_audio_video(vtemp, atemp, outpath)
        os.remove(vtemp)
        os.remove(atemp)
        return None
    def ymax (self, img=None, return_img=False, overwrite_imgs=False):
        if img is None:
            if hasattr(self, 'imgs'):
                img = self.imgs
            elif hasattr(self, 'img'):
                img = self.img
            else:
                raise ValueError("Provide 'img'.")
        if len(img.shape)==2:
            pos_img = image.ymax(img)
            self.ymaxpos = pos_img['pos']
            if return_img:
                if overwrite_imgs:
                    self.imgs = pos_img['img']
                else:
                    self.ymaximgs = pos_img['img']
        elif len(img.shape)==3:
            pos_img = image.parallelize(img, image.ymax)
            self.ymaxpos = [ i['pos'] for i in pos_img ]
            if return_img:
                if overwrite_imgs:
                    self.imgs = [ i['img'] for i in pos_img ]
                else:
                    self.ymaximgs = [ i['img'] for i in pos_img ]
        return None


