import pathlib
import re
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
import math
import os
import subprocess
import soundfile as sf
import pyper
import multiprocessing as mp
from tqdm import tqdm

class Files:
    def __init__ (self):
        self.reset_attr()

    def reset_attr (self):
        self.paths = None
        return None

    def find (self, directory, extension, recursive=False, inplace=False):
        if extension == '/':
            files = self.find_dirs(directory, recursive)
        elif extension == '*':
            files = self.find_all(directory, recursive)
        else:
            files = self.find_files(directory, extension, recursive)
        if inplace:
            self.paths = files
            return None
        else:
            return files

    def find_files (self, directory, extension, recursive=False):
        path = pathlib.Path(directory)
        if extension[0] != '.':
            extension = '.' + extension
        target = '**/*' + extension if recursive else '*' + extension
        files = path.glob(target)
        files = [ str(i) for i in files ]
        files = sorted(files)
        return files

    def find_dirs (self, directory, recursive=False):
        path = pathlib.Path(directory)
        target = '**/*' if recursive else '*'
        files = path.glob(target)
        files = [ str(i) for i in files if i.is_dir() ]
        files = sorted(files)
        return files

    def find_all (self, directory, recursive=False):
        path = pathlib.Path(directory)
        target = '**/*' if recursive else '*'
        files = path.glob(target)
        files = [ str(i) for i in files ]
        files = sorted(files)
        return files

    def replace (self, target, replacement, path=None, inplace=False, regex=False):
        if path is None:
            path = self.paths
        isstr = isinstance(path, str)
        if isstr:
            path = [path]
        if regex:
            newpath = [ re.sub(target, replacement, i) for i in path ]
        else:
            newpath = [ i.replace(target, replacement) for i in path ]
        if isstr:
            newpath = newpath[0]
        if inplace:
            self.paths = newpath
            newpath = None
        return newpath

    def name (self, path=None, inplace=False, drop_extension=False):
        if path is None:
            path = self.paths
        isstr = isinstance(path, str)
        if isstr:
            path = [path]
        path = [ pathlib.Path(i) for i in path ]
        if drop_extension:
            path = [ i.stem for i in path ]
        else:
            path = [ i.name for i in path ]
        if isstr:
            path = path[0]
        if inplace:
            self.names= path
            path = None
        return path

    def suffix (self, path=None, inplace=False, keep_period=True ):
        if path is None:
            path = self.paths
        isstr = isinstance(path, str)
        if isstr:
            path = [path]
        path = [ pathlib.Path(i) for i in path ]
        if keep_period:
            path = [ i.suffix for i in path ]
        else:
            path = [ i.suffix[1:] for i in path ]
        if isstr:
            path = path[0]
        if inplace:
            self.suffixes= path
            path = None
        return path
    
    def exist ( self, path=None, verbose=False ):
        if path is None:
            path = self.paths
        isstr = isinstance(path, str)
        if isstr:
            path = [path]
        path = [ pathlib.Path(i) for i in path ]
        errp = [ i for i in path if not (i.is_file() or i.is_dir()) ]
        res = not (len(errp)>0)
        if (not res) and verbose:
            print('The paths below do not exist.')
            for i in errp:
                print(i)
        return res

    def parent ( self, path=None, inplace=False):
        if path is None:
            path = self.paths
        isstr = isinstance(path, str)
        if isstr:
            path = [path]
        path = [ pathlib.Path(i) for i in path ]
        isdr = [ i.is_dir() for i in path ]
        path = [ i if j else i.parent for i,j in zip(path, isdr) ]
        path = [ str(i) for i in path ]
        if isstr:
            path = path[0]
        if inplace:
            self.paths= path
            path = None
        return path


class UltFiles (Files):
    def __init__(self, directory=''):
        if directory!='':
            self.set_paths(directory=directory)

    def set_paths (self, directory):
        self.wdir = directory
        self.paths = {}
        self.paths['txt']      = self.find(directory, '.txt')
        self.paths['ult']      = self.find(directory, '.ult')
        self.paths['wav']      = self.find(directory, '.wav')
        self.paths['textgrid'] = self.find(directory, '.TextGrid')
        self.paths['ustxt'] = [ i for i in self.paths['txt'] if 'US.txt' in i ]
        self.paths['txt']   = [ i for i in self.paths['txt'] if not 'US.txt' in i ]
        self.paths['txt']   = [ i for i in self.paths['txt'] if not '_Track' in i ]
        self.paths['phoneswithQ']      = self.find(directory, '.phoneswithQ')
        self.paths['words']      = self.find(directory, '.words')
    
    def is_all_set (self, verbose=False, exclude_empty_list=False):

        def clean_pathdict (path_dict):
            path_dict = { i:[self.name(k, drop_extension=True) for k in j] for i,j in path_dict.items() }
            path_dict = { i:[clean_name(k) for k in j] for i,j in path_dict.items() }
            return path_dict

        def clean_name ( nm ):
            nm = nm.split('.')
            tgts = ['_Track[0-9]+$', 'US$']
            for i in tgts:
                nm[0] = re.sub(i, '', nm[0])
            return '.'.join(nm)

        def type_diff (path_dict):
            sets = { i:set(j) for i,j in path_dict.items() }
            for i in sets.keys():
                for j in sets.keys():
                    diff = sets[i] - sets[j]
                    if len(diff)!=0:
                        for k in diff:
                            print('Filename "{}" exists only with .{}, but not with .{}.'.format(k,i,j))
        
        def check_ind (path_dict, verbose=False):
            kys = path_dict.keys()
            bln = len(path_dict[list(kys)[0]])
            allsame = True
            for ind in range(bln):
                tgt = {}
                for i in kys:
                    tgt.update({i:path_dict[i][ind]})
                if len(set(tgt.values()))!=1:
                    if verbose:
                        allsame=False
                        for j,k in tgt.items():
                            print('Filenames --> Not match.')
                            print('{}: {}'.format(j,k))
                    else:
                        return False
            return allsame

        tgt_dct = self.paths
        if exclude_empty_list:
            tgt_dct = { i:j for i,j in tgt_dct.items() if len(j)>0 }
        tgt_dct = clean_pathdict(tgt_dct)
        lens = [ len(i) for i in tgt_dct.values() ]
        if len(set(lens))!=1:
            allsame = False
            if verbose:
                print('Numbers of paths --> Not match.')
                type_diff(tgt_dct)
        else:
            if verbose:
                print('Numbers of paths --> OK.')
            sets = { i:set(j) for i,j in tgt_dct.items() }
            lens = [ len(i) for i in sets.values() ]
            if len(set(lens))!=1:
                allsame = False
                if verbose:
                    print('Numbers of unique paths --> Not match.')
                    type_diff(tgt_dct)
            else:
                if verbose:
                    print('Numbers of unique paths --> OK.')
                allsame = check_ind(tgt_dct, verbose)
                if allsame and verbose:
                    print('Filenames --> OK.')
        return allsame

    def dict_to_attr ( self, dct ):
        for i,j in dct.items():
            ky = i.lower()
            setattr(self, ky, j)

class Prompt (UltFiles):
    def __init__(self, path=None, encoding=None):
        self.reset_attr()
        if not path is None:
            self.read(path, encoding=encoding, inplace=True)
    def reset_attr(self):
        self.prompt, self.date, self.participant = ['']*3
    def read(self, path, encoding=None, inplace=False):
        Prompt.reset_attr(self)
        with open(path, "r", encoding=encoding) as f:
            content = f.readlines()
        content = [ i.rstrip('\n') for i in content ]
        content = [ i.rstrip(',') for i in content ]
        kys = ['prompt', 'date', 'participant']
        content = { i:j for i,j in zip(kys,content) }
        if inplace:
            self.dict_to_attr(content)
            content = None
        return content
    def to_alignment_txt ( self, prmptpath, wavpath):
        prompt = Prompt.read(self, prmptpath)
        prompt = prompt['prompt']
        prompt = prompt.split(' ')
        prompt = [ i + '\n' for i in prompt ]
        opath = wavpath.replace('.wav', '.txt')
        with open(opath, 'w') as f:
            f.writelines(prompt)
        return None

        

class UStxt (UltFiles):
    def __init__(self, path=None, encoding=None):
        self.reset_attr()
        if not path is None:
            self.read(path, encoding=None, inplace=True)
    def reset_attr(self):
        self.numvectors = None
        self.pixpervector = None
        self.framesize = None
        self.zerooffset = None
        self.bitsperpixel = None
        self.angle = None
        self.kind = None
        self.pixelspermm = None
        self.framespersec = None
        self.timeinsecsoffirstframe = None
    def read (self, path, encoding=None, inplace=False):
        UStxt.reset_attr(self)
        with open(path, "r", encoding=encoding) as f:
            content = f.readlines()
        content = [ i.rstrip('\n') for i in content ]
        content = [ i.split('=') for i in content ]
        content = { i[0]:i[1] for i in content }
        content = self.correct_dct_el(content)
        content['FrameSize'] = content['NumVectors'] * content['PixPerVector']
        if inplace:
            self.dict_to_attr(content)
            content = None
        return content
    def correct_dct_el (self, dct):
        kys = dct.keys()
        for i in kys:
            c_item = dct[i]
            isfloat = '.' in c_item
            try:
                dct[i] = float(c_item) if isfloat else int(c_item)
            except ValueError:
                pass
        return dct

class Ult (UltFiles):
    def __init__(self, path=None):
        self.reset_attr()
        if not path is None:
            self.read(path, inplace=True)
    def reset_attr(self):
        self.vector = None
    def read (self, path, inplace=False):
        Ult.reset_attr(self)
        vect = np.fromfile(open(path, "rb"), dtype=np.uint8)
        res_dct = { 'vector':vect }
        if inplace:
            self.dict_to_attr(res_dct)
            res_dct = None
        return res_dct

class UltAnalysis (Ult, UStxt, Prompt):
    def __init__ (self):
        pass

    def read (self, path, inplace=False):
        is_ustxt = bool(re.search('US\\.txt$', path))
        if is_ustxt:
            res_dct = UStxt.read(self, path)
        else:
            sfx = self.suffix(path)
            if sfx=='.txt':
                res_dct = Prompt.read(self, path)
            elif sfx=='.ult':
                res_dct = Ult.read(self, path)
            else:
                print('Filetype "{}" is not implemented yet.'.format(sfx))
                return None
        if inplace:
            self.dict_to_attr(res_dct)
            res_dct = None
        return res_dct

    def vec_to_pics (self, vector=None, inplace=False):
        if vector is None:
            vector = self.vector
        self._add_number_of_frames()
        img = vector.reshape(self.number_of_frames, self.numvectors, self.pixpervector)
        img  = np.rot90(img, axes=(1,2))
        img = self._inplace_img(img=img, inplace=inplace)
        return img

    def _add_number_of_frames (self):
        try:
            self.number_of_frames = self.vector.size // int(self.framesize)
        except AttributeError:
            print('No enough attributes found. Check if all the necessary files are loaded.')

    def _inplace_img (self, img, inplace):
        if inplace:
            self.img = img
            img = None
        return img

    def _inplace_df (self, df, inplace):
        if inplace:
            self.df = df
            df = None
        return df 

    def _format_imgs (self, img=None, rgb=False):
        img = self._getimg(img=img)
        typ = self._imgtype(img=img, rgb=rgb)
        if typ=='n2':
            img = [[img]]
        elif typ=='n3' or typ=='l2':
            img = [ [i] for i in img ]
        elif typ=='l3':
            img = [ [ j for j in i ] for i in img ]
        else:
            raise ValueError('Image type is not recognizable. Check the structure of the input image(s).')
        return img

    def _finish_imgs (self, img):
        if len(img)==1:
            img = img[0][0]
        else:
            if len(img[0])==1:
                img = [ j for i in img for j in i ]
                shp = [ i.shape for i in img ]
                if len(set(shp))==1:
                    img = np.stack(img)
        return img

    def _getimg (self, img=None):
        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None
        return img

    def _imgtype (self, img=None, rgb=False):
        img = self._getimg(img=img)
        islist = isinstance(img, list)
        isnump = isinstance(img, np.ndarray)
        if islist:
            if not isinstance(img[0], np.ndarray):
                raise ValueError("Elements in a list of images should be numpy.ndarray.")
            dims = len(img[0].shape)
            if dims==2:
                imgtype = 'l2'
            elif dims==3:
                imgtype = 'l3'
            else:
                raise ValueError("Provide each image in a list in the 2 or 3 dimensions.")
        elif isnump:
            dims = len(img.shape)
            if dims==2:
                imgtype = 'n2'
            elif dims==3:
                if rgb:
                    imgtype = 'n2'
                else:
                    imgtype = 'n3'
            else:
                raise ValueError("Images should be 2 or 3 dimensions.")
        else:
            raise ValueError("Provide image(s) as list or numpy.array.")
        return imgtype


class UltPicture (UltAnalysis):
    def __init__ (self):
        self.reset_attr()
        return None

    def reset_attr (self):
        self.img = None
        Files.reset_attr(self)
        Prompt.reset_attr(self)
        UStxt.reset_attr(self)
        Ult.reset_attr(self)
        return None

    def save_img (self, path, img=None):
        img = self._getimg(img=img)
        cv2.imwrite(path, img)
        return None

    def read_img (self, path, grayscale=True, inplace=False):
        img = cv2.imread(path, 0) if grayscale else cv2.imread(path)
        img = self._inplace_img(img=img, inplace=inplace)
        return img

    def flip (self, flip_direction=None, img=None, inplace=False, rgb=False):
        def __flip_drct ( img, flip_direction ):
            num_dim = len(img.shape)
            mod_dim = num_dim - 2
            if rgb:
                mod_dim=0
            if mod_dim<0 or mod_dim>1:
                raise ValueError('self.flip is implemented only for 2 or 3 dimensions.')
            ydim = 0 + mod_dim
            xdim = 1 + mod_dim
            if flip_direction is None:
                flip_direction = ( ydim, xdim )
            else:
                x_in = 'x' in flip_direction
                y_in = 'y' in flip_direction
                if x_in and y_in:
                    flip_direction = ( ydim, xdim )
                elif x_in and not y_in:
                    flip_direction = xdim
                elif not x_in and y_in:
                    flip_direction = ydim
                else:
                    raise ValueError('Provide directions of flipping as "x", "y", or "xy".')
            return flip_direction
        img = self._format_imgs(img=img, rgb=rgb)
        drct = __flip_drct(img=img[0][0], flip_direction=flip_direction)
        img = [ [ np.flip(j, drct) for j in i ] for i in img ]
        img = self._finish_imgs(img=img)
        img = self._inplace_img(img=img, inplace=inplace)
        return img

    def reduce_resolution (self, img=None, every_x=1, every_y=1, inplace=False):
        img = self._format_imgs(img=img)
        img = [ [ j[::every_y, ::every_x] for j in i ] for i in img ]
        img = self._finish_imgs(img=img)
        img = self._inplace_img(img=img, inplace=inplace)
        return img

    def resize (self, new_xy, img=None, inplace=False):
        img = self._format_imgs(img=img)
        img = [ [ cv2.resize(src=j, dsize=new_xy) for j in i ] for i in img ]
        img = self._finish_imgs(img=img)
        img = self._inplace_img(img=img, inplace=inplace)
        return img

    def to_square (self, img=None, glbl=False, inplace=False, rgb=False):
        img = self._format_imgs(img=img, rgb=rgb)

        def __squsize (img, rgb=False):
            num_dim = len(img.shape)
            if rgb:
                xlen = img.shape[1]
                ylen = img.shape[0]
            else:
                xlen = img.shape[-1]
                ylen = img.shape[-2]
            bigger = ylen if ylen >= xlen else xlen
            squsize = (bigger, bigger)
            return squsize

        def __resize_squ (img, rgb=False):
            squsize = __squsize(img=img, rgb=rgb)
            img = cv2.resize(src=img, dsize=squsize)
            return img

        if glbl:
            shp = [ j.shape for i in img for j in i ]
            ymx = max([ i[0] for i in shp ])
            xmx = max([ i[1] for i in shp ])
            bigger = ymx if ymx >= xmx else xmx
            squsize = (bigger, bigger)
            img = [ [ cv2.resize(src=j, dsize=squsize) for j in i ] for i in img ]

        else:
            img = [ [ __resize_squ(img=j, rgb=rgb) for j in i ] for i in img ]
        img = self._finish_imgs(img=img)
        img = self._inplace_img(img=img, inplace=inplace)
        return img

    def crop (self, crop_points, img=None, x_reverse=False, y_reverse=False, inplace=False, lineonly=False):
        img = self._format_imgs(img=img)
        img = [ [ self.__crop2d(crop_points, j, x_reverse, y_reverse, lineonly) for j in i ] for i in img ]
        img = self._finish_imgs(img=img)
        img = self._inplace_img(img=img, inplace=inplace)
        return img

    def __crop2d (self, crop_points, img=None, x_reverse=False, y_reverse=False, lineonly=False):
        ylen, xlen = img.shape
        xmin, xmax, ymin, ymax = crop_points
        xmin = 0 if xmin is None else xmin
        ymin = 0 if ymin is None else ymin
        xmax = xlen-1 if xmax is None else xmax
        ymax = ylen-1 if ymax is None else ymax
        if x_reverse:
            xmax_new = xlen - xmin
            xmin_new = xlen - xmax
            xmax = xmax_new - 1
            xmin = xmin_new - 1
        if y_reverse:
            ymax_new = ylen - ymin
            ymin_new = ylen - ymax
            ymax = ymax_new - 1
            ymin = ymin_new - 1
        if lineonly:
            img = img.astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img[ymin,:,:] = (0,0,255) 
            img[ymax,:,:] = (0,0,255) 
            img[:,xmin,:] = (0,0,255) 
            img[:,xmax,:] = (0,0,255) 
        else:
            img = img[ymin:(ymax+1), xmin:(xmax+1)]
        return img

        
    def fanshape (self, img=None, magnify=1, reserve=1800, bgcolor=255, inplace=False, verbose=False, force_general_param=False, numvectors=None, cores=1, progressbar=False):
        img = self._getimg(img=img)
        dimnum = len(img.shape)
        if dimnum==2:
            img = self.fanshape_2d(img, magnify, reserve, bgcolor, False, verbose, force_general_param, numvectors)
        elif dimnum==3:
            if img.shape[-1]==3:
                img = self.fanshape_2d(img, magnify, reserve, bgcolor, False, verbose, force_general_param, numvectors)
            else:
                img_fst = [ self.fanshape_2d(i, magnify, reserve, bgcolor, False, verbose, force_general_param, numvectors) for i in img[0:1]  ]
                if cores>1:
                    args = [ (i, magnify, reserve, bgcolor, False, False, force_general_param, numvectors)   for i in img[1:] ]
                    pool = mp.Pool(cores)
                    if progressbar:
                        img_rst = list(tqdm(pool.imap(self._par_fanshape, args), total=len(args), desc='Fanshape'))
                    else:
                        img_rst = pool.map(self._par_fanshape, args)
                else:
                    if progressbar:
                        img_rst = [ self.fanshape_2d(i, magnify, reserve, bgcolor, False, False, force_general_param, numvectors)   for i in tqdm(img[1:], desc='Fanshape') ]
                    else:
                        img_rst = [ self.fanshape_2d(i, magnify, reserve, bgcolor, False, False, force_general_param, numvectors)   for i in img[1:] ]
                img = img_fst + img_rst
        else:
            raise ValueError('self.fanshape is implemented only for a single or list of 2D grayscale images or a single RGB image.')

        img = self._inplace_img(img=img, inplace=inplace)
        return img

    def _par_fanshape(self, args):
        img, magnify, reserve, bgcolor, inplace, verbose, force_general_param, numvectors = args
        res = self.fanshape_2d(img=img, magnify=magnify, reserve=reserve, bgcolor=bgcolor, inplace=inplace, verbose=verbose, force_general_param=force_general_param, numvectors=numvectors)
        return res

    def fanshape_2d (self, img=None, magnify=1, reserve=1800, bgcolor=255, inplace=False, verbose=False, force_general_param=False, numvectors=None):

        def cart2pol(x, y):
            r = math.sqrt(x**2 + y**2)
            th = math.atan2(y, x)
            return r, th
        
        def ult_cart2pol(output_coordinates, origin, num_of_vectors, angle, zero_offset, pixels_per_mm, grayscale):
            (r, th) = cart2pol(output_coordinates[0] - origin[0],
                               output_coordinates[1] - origin[1])
            r *= pixels_per_mm
            cl = num_of_vectors // 2
            if grayscale:
                res = cl - ((th - np.pi / 2) / angle), r - zero_offset
            else:
                res = cl - ((th - np.pi / 2) / angle), r - zero_offset, output_coordinates[2]
            return res

        def unique_element_number(vec):
            try:
                aaa = [ tuple(i) for i in vec ]
            except TypeError:
                aaa = vec
            try:
                res = len(set(aaa))
            except TypeError:
                print('Warning: the input is not iterable')
                res = 1
            return res

        def trim_picture(pic_matrix):
            img = pic_matrix
            if len(img.shape)==2:
                unique_column = np.apply_along_axis(unique_element_number, 0, img)
                img = img[:,unique_column!=1]
                unique_row = np.apply_along_axis(unique_element_number, 1, img)
                img = img[unique_row!=1,:]
            elif len(img.shape)==3:
                unique_row = np.array([ unique_element_number(i) for i in img ])
                img = img[unique_row!=1,:,:]
                unique_column = np.array([ unique_element_number(img[:,i,:]) for i in range(img.shape[1]) ])
                img = img[:,unique_column!=1,:]
            return img

        nec_attrs = [ 'angle', 'zerooffset', 'pixpervector', 'numvectors' ]
        param_ok = all([ hasattr(self, i) for i in nec_attrs ]) if not force_general_param else False

        if param_ok:
            angle = self.angle
            zero_offset = self.zerooffset
            pixels_per_mm = self.pixelspermm
            number_of_vectors = self.numvectors if numvectors is None else numvectors
        else:
            img = cv2.resize(img, (500,500))
            angle = 0.0031
            zero_offset = 150
            pixels_per_mm = 2
            number_of_vectors = img.shape[0]

        pixels_per_mm = pixels_per_mm//magnify

        if verbose:
            ok_text = 'Parameters provided by an US.txt file.'
            no_text = 'Default parameter values applied.'
            param_text =  ok_text if param_ok else no_text
            print(param_text)
            print('angle: '          + str(angle))
            print('zero_offset: '    + str(zero_offset))
            print('pixels_per_mm: '  + str(pixels_per_mm)    + ' (Modified by the argument "magnify")')
            print('num_of_vectors: ' + str(number_of_vectors))

        img = self._getimg(img=img)
        img = np.rot90(img, 3)
        dimnum = len(img.shape)
        if dimnum==2:
            grayscale = True
        elif dimnum==3 and img.shape[-1]==3:
            grayscale = False
        else:
            raise ValueError('Dimensions are not 2. And it does not look like a RGB format, either.')

        if grayscale:
            output_shape = (int(reserve // pixels_per_mm), int( (reserve*0.80) // pixels_per_mm))
        else:
            output_shape = (int(reserve // pixels_per_mm), int( (reserve*0.80) // pixels_per_mm), 3)
        origin = (int(output_shape[0] // 2), 0)
        img = ndimage.geometric_transform(img,
                mapping=ult_cart2pol,
                output_shape=output_shape,
                order=2,
                cval=bgcolor,
                extra_keywords={
                    'origin': origin,
                    'num_of_vectors': number_of_vectors,
                    'angle': angle,
                    'zero_offset': zero_offset,
                    'pixels_per_mm': pixels_per_mm,
                    'grayscale': grayscale})
        img = trim_picture(img)
        img = np.rot90(img, 1)
        img = self._inplace_img(img=img, inplace=inplace)
        return img

    def average_img (self, imgs):
        def same_shape (imglist):
            shps = [ i.shape for i in imglist ]
            res = len(set(shps))==1
            return res
        if isinstance(imgs, list):
            if not same_shape(imgs):
                shps = [ i.shape for i in imgs ]
                dim0 = max([ i[0] for i in shps ])
                dim1 = max([ i[1] for i in shps ])
                imgs = [ self.resize(new_xy=(dim0, dim1), img=i) for i in imgs ]
            imgs = np.stack(imgs)
        dimnum = len(imgs.shape)
        if dimnum!=3:
            raise ValueError('Input image does not have 3 dimensions.')
        mean_img = np.mean(imgs, axis=0)
        mean_img = mean_img.astype(int)
        return mean_img

    def byitem (self, imgs):
        mean_img = self.average_img(imgs=imgs)
        return mean_img

    def to_video ( self, outpath='./video.avi', imgs=None, fps=None ):
        imgs = self._getimg(img=imgs)
        if fps is None:
            fps = self.framespersec if hasattr(self, 'framespersec') else 10
        else:
            fps = int(fps)
        if not isinstance(imgs, list):
            if isinstance(imgs, np.ndarray):
                imgs = [ i for i in imgs ]
            else:
                raise TypeError('Provide images as a list, where each element corresponds to one frame.')
        if len(imgs)==1:
            print('Warning: No video is produced because only an single image is provided. Use self.save_img for writing an image out.')
            return None
        img_shape = imgs[0].shape
        height = img_shape[0]
        width = img_shape[1]
        is_grayscale = len(img_shape)==2
        if is_grayscale:
            imgs = [ cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in imgs ]
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter(outpath, fourcc, fps, (width, height))
        for i in imgs:
            out.write(i)
        out.release()
        return None

    def add_direction (self, img=None, arrow=True, inplace=False, color=(0,0,0)):
        img = self._getimg(img=img)
        height = img.shape[0]
        width  = img.shape[1]
        hwdiff = width/height
        font   = cv2.FONT_HERSHEY_SIMPLEX
        fscale = width/500
        thick  = round(width*2/500)
        thick  = 1 if thick < 1 else thick
        pos_y  = round(height*90/100)
        pos_x  = round(width*5/100)
        img    = cv2.putText(img=img, text='Back', org=(pos_x, pos_y),
                fontFace=font, fontScale=fscale, color=color, thickness=thick,
                lineType=cv2.CV_8UC4, bottomLeftOrigin=False)
        pos_y  = round(height*90/100)
        pos_x  = round(width*77/100)
        img    = cv2.putText(img=img, text='Front', org=(pos_x, pos_y),
                fontFace=font, fontScale=fscale, color=color, thickness=thick,
                lineType=cv2.CV_8UC4, bottomLeftOrigin=False)
        
        if arrow:
            pos_y_start = round(height*88/100)
            pos_x_start = round(width*23/100)
            pos_y_end   = round(height*88/100)
            pos_x_end   = round(width*73/100)
            img         = cv2.arrowedLine(img=img,
                    pt1=(pos_x_start, pos_y_start), pt2=(pos_x_end, pos_y_end),
                    color=color, thickness=thick, line_type=cv2.CV_8UC4)
        if inplace:
            self.img = img
            return None
        else:
            return img

    def add_line ( self, img=None, pos_x=None, pos_y=None, thickness=1, color=(0,0,255), inplace=False):
        if thickness>0:
            thk = thickness//2
            thk = int(thk)
        else:
            raise ValueError('thickness must be positive.')
        img = self._getimg(img=img)
        img = img.astype('uint8')
        if len(img.shape)==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if not pos_x is None:
            img[:, pos_x-thk:pos_x+thk+1, :] = color
        if not pos_y is None:
            img[pos_y-thk:pos_y+thk+1, :, :] = color
        img = self._inplace_img(img=img, inplace=inplace)
        return img





class Alignment (Files):
    def run_aligner ( self, wavpath, aligner_home_path, alang='deu'):
        os.environ['LANG'] = 'C'
        os.environ['ALIGNERHOME'] = aligner_home_path
        os.environ['ALANG'] = alang
        os.environ['PATH'] = '{}/bin/{}:{}'.format(os.environ['ALIGNERHOME'], os.environ['ALANG'], os.environ['PATH'])
        cmds = ['Alignphones', 'Alignwords']
        for i in cmds:
            cmd = [i, wavpath]
            subprocess.call(cmd)
        return None
    
    def read_align_file ( self, path, inplace=False ):
        sfx = self.suffix(path)
        isp = sfx in ['.phonemic', '.phones', '.phoneswithQ']
        isw = sfx in ['.words']
        colns = ['end', 'segment'] if isp else ['end','word']
        dat = pd.read_csv(path, sep=' ', header=None, skiprows=[0], usecols=[0,2], names=colns)
        if inplace:
            atnm = 'segment' if isp else 'word'
            setattr(self, atnm, dat)
            dat = None
        return dat

    def format_align_files ( self ):
        tgts = [ 'segment', 'word' ]
        for i in tgts:
            try:
                df = self.__dict__[i]
            except KeyError:
                pass
            df['start'] = df['end'].shift(fill_value=0)
            self.__dict__[i] = df
        return None

    def comb_segm_word ( self, df_segment=None, df_word=None ):
        if df_segment is None:
            df_segment = self.segment
        if df_word is None:
            df_word = self.word
        def word_now (value, wrds):
            res = [ k for i,j,k in zip(wrds.start, wrds.end, wrds.word) if (value>i) and (value<=j) ]
            if len(res)!=1:
                raise ValueError('The provided value corresponds to more than one word.')
            return res[0]
        df_segment['word'] = [ word_now(i, df_word) for i in df_segment['end'] ]
        setattr(self, 'align_df', df_segment)
        return None

    def __wav_dur ( self, wavpath, inplace=False ):
        sound, rate = sf.read(wavpath)
        wavdur = len(sound)/rate
        if inplace:
            self.wavdur = wavdur
            wavdur = None
        return wavdur

    def to_textgrid ( self, wavpath, segment=None, word=None ):
        if segment is None:
            if hasattr(self, 'segment'):
                segment = self.segment
            else:
                raise ValueError('"segment" is None and the instance does not have it either.')
        if word is None:
            if hasattr(self, 'word'):
                word = self.word
            else:
                raise ValueError('"word" is None and the instance does not have it either.')
        if isinstance(segment, str):
            segment = self.read_align_file(path=segment)
        if isinstance(word, str):
            word = self.read_align_file(path=word)
        wavdur = self.__wav_dur(wavpath)
        tgtx = self.__textgrid_main(segment=segment, word=word, wavdur=wavdur)
        outpath = wavpath.replace('.wav','.TextGrid')
        with open(outpath, 'w') as f:
            f.writelines(tgtx)
        return None

    def __textgrid_main ( self, segment, word, wavdur ):
        hdtx = self.__textgrid_header(wavdur=wavdur)
        mtx_seg = self.__textgrid_middle(wavdur=wavdur, name='segments')
        btx_seg = self.__textgrid_body(df=segment, word_or_segment='segment')
        mtx_wrd = self.__textgrid_middle(wavdur=wavdur, name='words')
        btx_wrd = self.__textgrid_body(df=word, word_or_segment='word')
        comb = hdtx + mtx_seg + btx_seg + mtx_wrd + btx_wrd
        comb = [ i + '\n' for i in comb ]
        return comb

    def __four_spaces ( self ):
        return '        '

    def __textgrid_header ( self, wavdur ):
        hdtx = []
        hdtx.append('File type = "ooTextFile"')
        hdtx.append('Object class = "TextGrid"')
        hdtx.append('')
        hdtx.append('xmin = 0')
        hdtx.append('xmax = {}'.format(wavdur))
        hdtx.append('tiers? <exists> ')
        hdtx.append('size = 2')
        hdtx.append('item []:')
        return hdtx

    def __textgrid_middle ( self, wavdur, name ):
        tab = self.__four_spaces()
        mdtx = []
        mdtx.append('{0}item [1]:'.format(tab))
        mdtx.append('{0}{0}class = "IntervalTier"'.format(tab))
        mdtx.append('{0}{0}name = "{1}"'.format(tab,name))
        mdtx.append('{0}{0}xmin = 0'.format(tab))
        mdtx.append('{0}{0}xmax = {1}'.format(tab,wavdur))
        return mdtx

    def __textgrid_body ( self, df, word_or_segment):
        ctype = word_or_segment
        tab = self.__four_spaces()
        res = [ '{0}{0}intervals: size = {1}'.format(tab,len(df)) ]
        for i in range(len(df)):
            iv = i+1
            st = df['start'][i]
            ed = df['end'][i]
            wd = df[ctype][i]
            res.append('{0}{0}intervals [{1}]:'.format(tab,iv))
            res.append('{0}{0}{0}xmin = {1}'.format(tab,st))
            res.append('{0}{0}{0}xmax = {1}'.format(tab,ed))
            res.append('{0}{0}{0}text = "{1}"'.format(tab,wd))
        return res

    def prepare_audio ( self, audiopath, outpath, ff=None ):
        if ff is None:
            try:
                ff = self.timeinsecsoffirstframe
            except AttributeError:
                ff = 0
                print('WARNING: Time in the first frame is not found. Ultrasound images begin to be recorded usually AFTER the audio recording started. Check if the beginnings of the audio and video really match.')
        cmd = ['sox', audiopath, outpath, 'trim', str(ff), 'rate', '16000']
        subprocess.call(cmd)
        return None


class UltDf (UltAnalysis):
    def __init__ (self, obj=None):
        self.reset_attr()
        if not obj is None:
            self.__dict__ = obj.__dict__.copy()
        return None

    def reset_attr (self):
        self.df = None
        Files.reset_attr(self)
        Prompt.reset_attr(self)
        UStxt.reset_attr(self)
        Ult.reset_attr(self)
        return None

    def img_to_df (self, img=None, inplace=False, reverse=None, combine=True, add_time=False, fps=None, norm_frame=False, norm_df=False):

        img = self._format_imgs(img=img)

        if len(img[0])>1:
            cnt = 0
            for inda,i in enumerate(img):
                for indb,j in enumerate(i):
                    img[inda][indb] = self.__to_df_2d(img=j, frame_id=cnt, reverse=reverse)
                    cnt += 1
        else:
            img = [ [ self.__to_df_2d(img=j, frame_id=frm, reverse=reverse) for j in i ] for frm,i in enumerate(img) ]

        dfs = [ j for i in img for j in i ]

        if isinstance(norm_frame, bool):
            if norm_frame:
                dfs = [ self.normalize(df=i, coln=None) for i in dfs ]
        else:
            dfs = [ self.normalize(df=i, coln=norm_frame) for i in dfs ]

        if combine:
            dfs = pd.concat(dfs, ignore_index=True)

        if add_time:
            dfs = self.__addtime(df=dfs, fps=fps)

        if isinstance(norm_df, bool):
            if norm_df:
                dfs = self.normalize(df=dfs, coln=None)
        else:
            dfs = self.normalize(df=dfs, coln=norm_df)

        if inplace:
            self.df = dfs
            return None
        else:
            return dfs

    def __to_df_2d (self, img=None, frame_id=None, reverse=None):
        if not reverse is None:
            if 'x' in reverse:
                img = img[:, ::-1]
            if 'y' in reverse:
                img = img[::-1, :]
        ult  = img.flatten()
        xlen = img.shape[1]
        ylen = img.shape[0]
        xxx = np.array(list(range(xlen))*ylen)
        yyy = np.repeat(np.arange(ylen), xlen)
        df = pd.DataFrame({'brightness':ult, 'x':xxx, 'y':yyy})
        if not frame_id is None:
            df['frame'] = frame_id
        return df

    def __addtime (self, df=None, fps=None):
        df = self.__getdf(df=df)
        if fps is None:
            if hasattr(self, 'framespersec'):
                fps = self.framespersec
            else:
                raise AttributeError('fps (self.framespersec) not found.')

        def __calctime (df, fps):
            df['time'] = df['frame'] * (1/fps)
            return df

        if isinstance(df, list):
            df = [ __calctime(i, fps) for i in df ]
        else:
            df = __calctime(df, fps)
        return df

    def normalize_vec (self, vec):
        if isinstance(vec[0],str):
            res = vec
        else:
            if isinstance(vec, list):
                vec = pd.Series(vec)
            res = (vec - vec.min()) / (vec.max() - vec.min())
        return res

    def normalize (self, df=None, coln=None):
        df = self.__getdf(df=df)
        if coln is None:
            df = df.apply(self.normalize_vec)
        else:
            df = df.apply(lambda x: self.normalize_vec(x) if x.name in coln else x)
        return df

    def save_dataframe (self, outpath, df=None):
        df = self.__getdf(df=df)
        df.to_csv(outpath, sep='\t', header=True, index=False)
        return None

    def __getdf (self, df=None):
        if df is None:
            try:
                df = self.df
            except AttributeError:
                print('Error: Dataframe not found')
                return None
        return df

    def fit_gam (self, df, formula, outdir=None):
        if outdir is None:
            outdir = self.wdir
        if isinstance(df, pd.DataFrame):
            outpath_df = outdir + '/df.gz'
            self.save_dataframe(outpath_df, df)
            df = outpath_df
        if not self.exist(formula):
            outpath_frml = outdir + '/formula.txt'
            with open(outpath_frml, 'w') as f:
                f.write(formula)
            formula = outpath_frml

        r = pyper.R()
        r('ready_mgcv = require(mgcv)')
        r('ready_datatable = require(data.table)')
        ready_mgcv = r('ready_mgcv').split(' ')[1].strip()
        ready_datatable = r('ready_datatable').split(' ')[1].strip()

        if all([ready_mgcv=='TRUE', ready_datatable=='TRUE']):
            r('dat = fread("{}", sep="\t")'.format(df))
            r('frml = scan("{}", character())'.format(formula))
            r('frml = paste(frml, collapse=" ")')
            r('frml = gsub(", *data *= *[^,)]+", ", data=dat", frml)')
            r('eval(parse(text=sprintf("mdl = %s", frml)))')
            r('save(list=c("mdl"), file="{}")'.format(outdir+'/model.Rdata'))
            print('Fitted model is produced in the specified directory or in the directory set by self.set_paths() under the name "model.Rdata"')
        else:
            raise ImportError('You need to install mgcv and data.table in R first.')
        return None

    def integrate_segments ( self, phoneswithQ_path=None, words_path=None, df=None, rmvnoise=False, inplace=False ):
        df = self.__getdf(df=df)
        phone_exist = self.exist(phoneswithQ_path)
        words_exist = self.exist(words_path)
        if phone_exist and words_exist:
            align = self.__formatted_segment_df(phoneswithQ_path, words_path)
            df['segment'] = ''
            df['word'] = ''
            for i in align.index:
                cst = align.loc[i,'start']
                ced = align.loc[i,'end']
                csg = align.loc[i,'segment']
                cwd = align.loc[i,'word']
                df.loc[ (df.time>cst) & (df.time<ced), 'segment'] = csg
                df.loc[ (df.time>cst) & (df.time<ced), 'word'] = cwd
            if rmvnoise:
                df = self.rmv_noise(df=df)
            df = self._inplace_df(df=df, inplace=inplace)
        else:
            raise FileNotFoundError('*.phoneswithQ or *.words does not exist.')
        return df

    def __formatted_segment_df ( self, phoneswithQ_path=None, words_path=None ):
        aln = Alignment()
        aln.read_align_file(phoneswithQ_path, inplace=True)
        aln.read_align_file(words_path, inplace=True)
        aln.format_align_files()
        aln.comb_segm_word()
        return aln.align_df

    def rmv_noise ( self, colnames=['segment','word'], df=None, noise=None, inplace=False):
        df = self.__getdf(df=df)
        if noise is None:
            noise = ['_p:_','<P>','']
        for i in colnames:
            df = df.loc[~df[i].isin(noise),]
        df = self._inplace_df(df=df, inplace=inplace)
        return df

    def integrate_spline_values ( self, df, splval ):
        df = self.__getdf(df=df)
        splval['x'] = splval.pop('index')
        splval['y_spline'] = splval.pop('fitted_values')
        spl = pd.DataFrame(splval)
        df = pd.merge(df, spl, on='x', how='left')
        return df

