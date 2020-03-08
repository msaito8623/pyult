import pathlib
import re
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
import math

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

class UltFiles (Files):
    def __init__(self, directory=''):
        if directory!='':
            self.set_paths(directory=directory)

    def set_paths (self, directory):
        self.paths = {}
        self.paths['txt']      = self.find(directory, '.txt')
        self.paths['ult']      = self.find(directory, '.ult')
        self.paths['wav']      = self.find(directory, '.wav')
        self.paths['textgrid'] = self.find(directory, '.TextGrid')
        self.paths['ustxt'] = [ i for i in self.paths['txt'] if 'US.txt' in i ]
        self.paths['txt']   = [ i for i in self.paths['txt'] if not 'US.txt' in i ]
    
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
        self.reset_attr()
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
        self.reset_attr()
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
        self.reset_attr()
        vect = np.fromfile(open(path, "rb"), dtype=np.uint8)
        res_dct = { 'vector':vect }
        if inplace:
            self.dict_to_attr(res_dct)
            res_dct = None
        return res_dct

class UltPicture (Ult, UStxt, Prompt):
    def __init__ (self):
        self.reset_attr()
        return None

    def reset_attr (self):
        self.img = None
        self.df = None
        return None

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

    def add_number_of_frames (self):
        try:
            self.number_of_frames = self.vector.size // int(self.framesize)
        except AttributeError:
            print('No enough attributes found. Check if all the necessary files are loaded.')

    def vec_to_pics (self, vector=None, inplace=False):
        if vector is None:
            vector = self.vector
        self.add_number_of_frames()
        img = vector.reshape(self.number_of_frames, self.numvectors, self.pixpervector)
        img  = np.rot90(img, axes=(1,2))
        if inplace:
            self.img = img
            img = None
        return img

    def save_img (self, path, img=None):
        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None
        cv2.imwrite(path, img)
        return None

    def read_img (self, path, grayscale=True, inplace=False):
        img = cv2.imread(path, 0) if grayscale else cv2.imread(path)
        if inplace:
            self.img = img
            return None
        else:
            return img

    def flip (self, flip_direction=None, img=None, inplace=False):
        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None
        num_dim = len(img.shape)
        mod_dim = num_dim - 2
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
        img = np.flip(img, flip_direction)
        if inplace:
            self.img = img
            img = None
        return img

    def reduce_resolution (self, img=None, every_x=1, every_y=1, inplace=False):
        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None
        num_dim = len(img.shape)
        if num_dim==2:
            img = img[::every_y, ::every_x]
        elif num_dim==3:
            img = img[:,::every_y, ::every_x]
        else:
            raise ValueError('self.reduce_resolution is implemented only for 2 or 3 dimensions.')
        if inplace:
            self.img = img
            return None
        else:
            return img

    def resize (self, new_xy, img=None, inplace=False):
        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None
        num_dim = len(img.shape)
        if num_dim==2:
            img = cv2.resize(img, new_xy)
        elif num_dim==3:
            img = [ cv2.resize(i, new_xy) for i in img ]
        else:
            raise ValueError('self.resize is implemented only for 2 or 3 dimensions.')
        if inplace:
            self.img = img
            return None
        else:
            return img

    def to_square (self, img=None, inplace=False):
        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None
        num_dim = len(img.shape)
        if not num_dim in [2,3] :
            raise ValueError('self.to_square is implemented only for 2 or 3 dimensions.')
        xlen = img.shape[-1]
        ylen = img.shape[-2]
        bigger = ylen if ylen >= xlen else xlen
        newxy= (bigger, bigger)
        img = self.resize(newxy, img)
        if inplace:
            self.img = img
            return None
        else:
            return img

    def crop (self, crop_points, img=None, x_reverse=False, y_reverse=False, inplace=False, lineonly=False):
        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None
        num_dim = len(img.shape)
        if num_dim==2:
            img = self.crop_2d(crop_points, img, x_reverse, y_reverse, False, lineonly)
        elif num_dim==3:
            img = self.crop_3d(crop_points, img, x_reverse, y_reverse, False, lineonly)
        else:
            raise ValueError('self.crop is implemented only for 2 or 3 dimensions.')
        if inplace:
            self.img = img
            return None
        else:
            return img

    def crop_3d (self, crop_points, img=None, x_reverse=False, y_reverse=False, inplace=False, lineonly=False):
        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None
        img = [ self.crop_2d(crop_points, i, x_reverse, y_reverse, inplace, lineonly) for i in img ]
        return img

    def crop_2d (self, crop_points, img=None, x_reverse=False, y_reverse=False, inplace=False, lineonly=False):
        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None
        ylen, xlen = img.shape
        xmin, xmax, ymin, ymax = crop_points
        xmin = 0 if xmin is None else xmin
        ymin = 0 if ymin is None else ymin
        xmax = xlen-1 if xmax is None else xmax
        ymax = ylen-1 if ymax is None else ymax
        if x_reverse:
            xmax_new = xlen - xmin
            xmin_new = xlen - xmax
            xmax = xmax_new
            xmin = xmin_new
        if y_reverse:
            ymax_new = ylen - ymin
            ymin_new = ylen - ymax
            ymax = ymax_new
            ymin = ymin_new
        if lineonly:
            img = img.astype('uint8')
            if len(img.shape)==2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            ymin = ymin-1 if ymin>0 else 0
            xmin = xmin-1 if xmin>0 else 0
            img[ymin,:,:] = (0,0,255) 
            img[ymax,:,:] = (0,0,255) 
            img[:,xmin,:] = (0,0,255) 
            img[:,xmax,:] = (0,0,255) 
        else:
            img = img[ymin:ymax, xmin:xmax]
        if inplace:
            self.img = img
            return None
        else:
            return img

        
    def to_df (self, img=None, inplace=False, reverse=None, combine=False, add_time=False, fps=None, norm_frame=False, norm_df=False):

        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None

        isndarray = isinstance(img, np.ndarray)
        islist = isinstance(img, list)

        if isndarray:
            num_dim = len(img.shape)
        elif islist:
            num_dim=3
        else:
            raise ValueError('Input image(s) should be numpy.ndarray or list')

        if num_dim==2:
            df = self.to_df_2d(img, False, reverse)
        elif num_dim==3:
            df = self.to_df_3d(img, False, reverse)
        else:
            raise ValueError('self.to_df is implemented only for 2 or 3 dimensions.')

        if isinstance(norm_frame, bool):
            if norm_frame:
                df = [ self.normalize(df=i, coln=None) for i in df ]
        else:
            df = [ self.normalize(df=i, coln=norm_frame) for i in df ]

        if combine:
            if num_dim==2:
                df = [df]
            df = pd.concat(df, ignore_index=True)

        if add_time:
            if fps is None:
                if hasattr(self, 'framespersec'):
                    fps = self.framespersec
                else:
                    raise AttributeError('fps (self.framespersec) not found.')
            df = self.add_time(df=df, fps=fps)

        if isinstance(norm_df, bool):
            if norm_df:
                df = self.normalize(df=df, coln=None)
        else:
            df = self.normalize(df=df, coln=norm_df)

        if inplace:
            self.df = df
            return None
        else:
            return df

    def to_df_3d (self, img=None, inplace=False, reverse=None):
        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None
        def temp ( img, reverse, ind ):
            c_df = self.to_df_2d(img, False, reverse)
            c_df['frame'] = ind
            return c_df
        img = [ temp(j, reverse, i) for i,j in enumerate(img) ]
        return img

    def to_df_2d (self, img=None, inplace=False, reverse=None):
        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None
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
        if inplace:
            self.df = df
            return None
        else:
            return df

    def add_time (self, df=None, fps=None, inplace=False):
        if df is None:
            if hasattr(self, 'df'):
                df = self.df
            else:
                raise AttributeError('Dataframe not found.')
        if fps is None:
            if hasattr(self, 'framespersec'):
                fps = self.framespersec
            else:
                raise AttributeError('fps (self.framespersec) not found.')
        df['time'] = df['frame'] * (1/fps)
        if inplace:
            self.df = df
            df = None
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
        if df is None:
            if hasattr(self, 'df'):
                df = self.df
            else:
                raise AttributeError('Dataframe not found.')
        if coln is None:
            df = df.apply(self.normalize_vec)
        else:
            df = df.apply(lambda x: self.normalize_vec(x) if x.name in coln else x)
        return df

    def save_dataframe (self, outpath, df=None):
        if df is None:
            try:
                df = self.df
            except AttributeError:
                print('Error: Dataframe not found')
                return None
        df.to_csv(outpath, sep='\t', header=True, index=False)
        return None

    def fanshape (self, img=None, magnify=1, reserve=1800, bgcolor=255, inplace=False, verbose=False):
        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None

        dimnum = len(img.shape)
        if dimnum==2:
            img = self.fanshape_2d(img, magnify, reserve, bgcolor, False, verbose)
        elif dimnum==3:
            if img.shape[-1]==3:
                img = self.fanshape_2d(img, magnify, reserve, bgcolor, False, verbose)
            else:
                img_fst = [ self.fanshape_2d(i, magnify, reserve, bgcolor, False, verbose) for i in img[0:1]  ]
                img_rst = [ self.fanshape_2d(i, magnify, reserve, bgcolor, False, False)   for i in img[1:] ]
                img = img_fst + img_rst
        else:
            raise ValueError('self.fanshape is implemented only for a single or list of 2D grayscale images or a single RGB image.')

        if inplace:
            self.img = img
            return None
        else:
            return img
        return img

    def fanshape_2d (self, img=None, magnify=1, reserve=1800, bgcolor=255, inplace=False, verbose=False):

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

        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None

        img = np.rot90(img, 3)

        nec_attrs = [ 'angle', 'zerooffset', 'pixpervector', 'numvectors' ]
        param_ok = all([ hasattr(self, i) for i in nec_attrs ])

        if param_ok:
            angle = self.angle
            zero_offset = self.zerooffset
            pixels_per_mm = self.pixelspermm
            number_of_vectors = self.numvectors
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
        if inplace:
            self.img = img
            return None
        else:
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
        if imgs is None:
            try:
                imgs = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None
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
        if img is None:
            try:
                img = self.img
            except AttributeError:
                print('Error: Image matrix not found')
                return None
    
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

