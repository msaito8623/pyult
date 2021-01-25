import numpy as np
import cv2
import pyult.ncspline as spl
from scipy import ndimage
import math
from tqdm import tqdm
import subprocess
import warnings
from pathlib import Path
from multiprocessing import Pool

def read_img (path, grayscale=True):
    flg = 0 if grayscale else 1
    img = cv2.imread(path,flg)
    return img

def save_img (outpath, img):
    shp = img.shape
    iln = len(shp)
    if (iln==2) or (iln==3 and shp[-1]==3):
        cv2.imwrite(outpath, img)
    elif (iln==3 and shp[-1]!=3) or (iln==4):
        opath = Path(outpath)
        prn = opath.parent
        stm = opath.stem
        opaths = [ '{}/{}{:03d}.png'.format(prn,stm,i) for i in range(len(img)) ]
        for i,j in zip(opaths,img):
            cv2.imwrite(i,j)
    else:
        raise ValueError('Image dimension is invalid.')
    return None

def vec_to_imgs (vector, number_of_vectors, pixel_per_vector):
    """
    Convert an ultrasound vector into a series of images.
    """
    frame_size = number_of_vectors * pixel_per_vector
    number_of_frames = vector.size // int(frame_size)
    imgs = vector.reshape(number_of_frames, number_of_vectors, pixel_per_vector)
    imgs = np.rot90(imgs, axes=(1,2))
    return imgs

def crop (imgs, crop_points):
    """
    Crop every frame in a series of images according to crop_points.
    """
    def __crop2d (img, crop_points):
        ylen, xlen = img.shape
        xmin, xmax, ymin, ymax = crop_points
        xmin = 0 if xmin is None else xmin
        ymin = 0 if ymin is None else ymin
        xmax = xlen-1 if xmax is None else xmax
        ymax = ylen-1 if ymax is None else ymax
        img = img[ymin:(ymax+1), xmin:(xmax+1)]
        return img
    first_frame = imgs[0] if len(imgs.shape)==3 else imgs
    if not crop_points is None:
        if isinstance(crop_points, str):
            crop_points = crop_points.split(',')
            defaults = [0, first_frame.shape[1], 0, first_frame.shape[0]]
            crop_points = [ defaults[i] if j=='None' else j for i,j in enumerate(crop_points) ]
            crop_points = [ int(i) for i in crop_points if i!='' ]
            crop_points = tuple(crop_points)
        if len(imgs.shape)==3:
            imgs = [ __crop2d(i, crop_points) for i in imgs ]
        elif len(imgs.shape)==2:
            imgs = __crop2d(imgs, crop_points)
        else:
            raise ValueError('Shapes of images are not compatible: 2 or 3 is acceptable.')
    imgs = np.array(imgs)
    return imgs

def add_crop_line (imgs, crop_points):
    """
    Highlight cropping points of the four sides by red lines.
    """
    def __crop2d (img, crop_points):
        ylen, xlen = img.shape
        xmin, xmax, ymin, ymax = crop_points
        xmin = 0 if xmin is None else xmin
        ymin = 0 if ymin is None else ymin
        xmax = xlen-1 if xmax is None else xmax
        ymax = ylen-1 if ymax is None else ymax
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img[ymin,:,:] = (0,0,255) 
        img[ymax,:,:] = (0,0,255) 
        img[:,xmin,:] = (0,0,255) 
        img[:,xmax,:] = (0,0,255) 
        return img
    if not crop_points is None:
        imgs = [ __crop2d(i, crop_points) for i in imgs ]
    imgs = np.array(imgs)
    return imgs

def flip (imgs, direct):
    if not direct is None:
        if direct=='x':
            direct = 1
        elif direct=='y':
            direct = 0
        elif direct=='xy':
            direct = (0,1)
        else:
            raise ValueError('The second argument "direct" should be "x", "y", or "xy".')
        if len(imgs.shape)==3:
            imgs = [ np.flip(i, direct) for i in imgs ]
        elif len(imgs.shape)==2:
            imgs = np.flip(imgs, direct)
        else:
            raise ValueError('Shapes of images are not compatible: 2 or 3 is acceptable.')
    imgs = np.array(imgs)
    return imgs

def reduce_y (imgs, every_nth):
    if not every_nth is None:
        if isinstance(every_nth, str):
            every_nth = int(every_nth)
        if len(imgs.shape)==3:
            imgs = [ i[::every_nth, :] for i in imgs ]
        elif len(imgs.shape)==2:
            imgs = imgs[::every_nth, :]
        else:
            raise ValueError('Shapes of images are not compatible: 2 or 3 is acceptable.')
    imgs = np.array(imgs)
    return imgs

### Functions for fitting spline curves (FROM HERE) ###
def fit_spline (imgs, return_values=False, return_imgs=True):
    imgshape = len(imgs.shape)
    if imgshape==3:
        imgs = [ fit_spline_2d(i, return_values, return_imgs) for i in imgs ]
    elif imgshape==2:
        imgs = fit_spline_2d(imgs, return_values, return_imgs)
    if return_imgs and return_values:
        if imgshape==3:
            images = [ i['image']         for i in imgs ]
            ftvs   = [ None if i is None else i['fitted_values'] for i in imgs ]
        elif imgshape==2:
            images = imgs['image']
            ftvs = imgs['fitted_values']
        ret = {'images':np.array(images), 'fitted_values':ftvs}
    elif (not return_imgs) and return_values:
        ret = imgs
    elif return_imgs and (not return_values):
        ret = np.array(imgs)
    else:
        ret = None
    return ret

def fit_spline_2d (img, return_values=False, return_imgs=True):
    ftv = get_fitted_values(img)

    if ftv is None:
        print('WARNING: Something is wrong with the input image. A spline curve could not be fitted. The input image is returned without a spline curve.')
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        yyy = ftv['fitted_values']
        xxx = ftv['index']
        if return_imgs:
            img = img.astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for xpos,ypos in zip(xxx, yyy):
                if not ypos is None:
                    img[ypos-2:ypos+2, xpos-2:xpos+2, :] = (0,0,255)

    if return_imgs and return_values:
        ret = {'image':img, 'fitted_values':ftv}
    elif (not return_imgs) and return_values:
        ret = ftv
    elif return_imgs and (not return_values):
        ret = img
    else:
        ret = None
    return ret

def get_fitted_values (img):
    pks = [ spl.fit_ncspline(img[:,i]) for i in range(img.shape[1])]
    pks = [ peaks(i) for i in pks ]
    pks = { i:j for i,j in enumerate(pks) }
    aaa = len(pks)//4 # allowed_range_x
    bbb = np.arange(aaa, len(pks)-aaa)
    selpks = { i:j for i,j in pks.items() if i in bbb }
    aaa = max([ i['peak_ratio'] for i in selpks.values() ])
    bbb = [ i['peak_ratio']==0 for i in selpks.values() ]
    if aaa==0 and all(bbb):
        trpk = { i:j for i,j in selpks.items() if i==len(selpks)//2 }
    else:
        trpk = { i:j for i,j in selpks.items() if j['peak_ratio']==aaa }
    trpk_id = list(trpk.keys())[0]
    def max_pos_val ( dct ):
        pos = dct['peak_vals'].argmax()
        mxps = dct['peak_poses'][pos]
        mxvl = dct['peak_vals'][pos]
        return { 'max_pos':mxps, 'max_val':mxvl }
    mxpv = max_pos_val(trpk[trpk_id])
    trpk[trpk_id]['peak_poses'] = mxpv['max_pos']
    trpk[trpk_id]['peak_vals'] = mxpv['max_val']
    def nearest_pos( current_pos, candidates, threshold=10000 ):
        if len(candidates)==0:
            pos = None
        else:
            diff = abs(candidates - current_pos)
            val = diff.min()
            pos = diff.argmin()
            pos = pos if val <= threshold else None
        return pos
    threshold = img.shape[0]/10
    threshold_val = 0.5
    cpos = trpk[trpk_id]['peak_poses']
    cval = trpk[trpk_id]['peak_vals']
    for i in range(trpk_id+1, max(pks.keys())+1):
        cand = pks[i]['peak_poses']
        tpos = nearest_pos(cpos, cand, threshold)
        if (tpos is None) or (pks[i]['peak_vals'][tpos]/cval < threshold_val):
            pks[i]['peak_poses'] = None
            pks[i]['peak_vals']  = None
        else:
            pks[i]['peak_poses'] = pks[i]['peak_poses'][tpos]
            pks[i]['peak_vals']  = pks[i]['peak_vals'][tpos]
            cpos = pks[i]['peak_poses']
            cval = pks[i]['peak_vals']
    cpos = trpk[trpk_id]['peak_poses']
    cval = trpk[trpk_id]['peak_vals']
    for i in range(trpk_id-1, min(pks.keys())-1, -1):
        cand = pks[i]['peak_poses']
        tpos = nearest_pos(cpos, cand, threshold)
        if (tpos is None) or (pks[i]['peak_vals'][tpos]/cval < threshold_val):
            pks[i]['peak_poses'] = None
            pks[i]['peak_vals']  = None
        else:
            pks[i]['peak_poses'] = pks[i]['peak_poses'][tpos]
            pks[i]['peak_vals']  = pks[i]['peak_vals'][tpos]
            cpos = pks[i]['peak_poses']
            cval = pks[i]['peak_vals']
    xy0 = { i:j['peak_poses'] for i,j in pks.items() }
    xy1 = { i:j for i,j in xy0.items() if not j is None }
    if len(xy1)>1:
        y = np.array(list(xy1.values()))
        x = np.array(list(xy1.keys()))
        nxmin = np.array(list(xy1.keys())).astype(int).min()
        nxmax = np.array(list(xy1.keys())).astype(int).max()
        nx = np.arange(nxmin, nxmax+1)
        ftv = spl.fit_ncspline(y=y, x=x, knots=10, pred_x=nx)
        ftv = ftv.round().astype(int)
        ftv_dict = {'index':nx, 'fitted_values':ftv}
    else:
        ftv_dict = None
    return ftv_dict

def peaks ( vect, howmany=3 ):
    f0 = vect
    d1 = deriv_discrete(f0)
    d2 = deriv_discrete(d1)
    d1_0 = np.where(d1==0)[0]
    def clean_d1_0 ( d1_0, approx=False ):
        def nearest (d1_0):
            def __cross_zero ( nums ):
                return (nums[0]>0 and nums[1]<0) or (nums[0]<0 and nums[1]>0)
            aaa = [ __cross_zero(d1[i:(i+2)]) for i in range(len(d1)-1) ]
            aaa = np.array(list(aaa) + [False])
            d1_0 = np.where(aaa)[0]
            return d1_0
        if approx or len(d1_0)==0:
            d1_0 = nearest(d1_0)
        try:
            aaa = d1_0 + 1
            bbb = np.concatenate([d1_0[1:],np.arange(1)])
            rmpos = np.where(aaa==bbb)[0]+1
            d1_0 = np.delete(d1_0, rmpos)
        except TypeError:
            pass
        return d1_0
    ccc = clean_d1_0(d1_0)
    peak_poses = ccc[d2[ccc]<0]
    if len(peak_poses)<=1:
        ccc = clean_d1_0(d1_0, approx=True)
        peak_poses = ccc[d2[ccc]<0]
    peak_vals = f0[peak_poses]
    allowed_range_y = 2/4
    peak_dict = { i:j for i,j in zip(peak_poses, peak_vals) if i <= len(f0)*allowed_range_y }
    max_vals = np.sort(np.array(list(peak_dict.values())))[::-1][:howmany]
    peak_dict = { i:j for i,j in peak_dict.items() if j in max_vals }
    peak_poses = np.array(list(peak_dict.keys()))
    peak_vals = np.array(list(peak_dict.values()))
    aaa = np.sort(peak_vals)[::-1][:2]
    try:
        peak_ratio = aaa[0]/aaa[1]
    except IndexError:
        peak_ratio = 0
    return {'peak_poses':peak_poses, 'peak_vals':peak_vals, 'peak_ratio':peak_ratio}

def deriv_discrete ( vect ):
    """
    Approximate the first derivative values for each element of a vector
    (excluding its endpoints), using the finite difference. Note
    that this function assumes that the values of the input vector
    are aligned with the same interval with each other.

    Parameters
    ----------
    vect : numpy.ndarray() or list()
        One-dimensional array of the input vector, from which
        first derivative values are approximated.

    Returns:
    ----------
    drvs : numpy.ndarray or list
        Approximated first derivatives, using the finite difference.
    """
    def __deriv ( vals ):
        if len(vals)!=3:
            raise ValueError('Length does not equals to 3.')
        return vals[-1] - vals[0]
    isnp = isinstance(vect, np.ndarray)
    drvs = [ __deriv(vect[(i-1):(i+2)]) for i in range(1,len(vect)-1) ]
    drvs = [drvs[0]] + drvs + [drvs[-1]]
    if isnp:
        drvs = np.array(drvs)
    return drvs

### Functions for fitting spline curves (UNTIL HERE) ###

def to_square (imgs):
    if len(imgs.shape)==4:
        shapes = [ i.shape for i in imgs ]
        ymx = max([ i[0] for i in shapes ])
        xmx = max([ i[1] for i in shapes ])
        bigger = ymx if ymx >= xmx else xmx
        squsize = (bigger, bigger)
        imgs = [ cv2.resize(src=i, dsize=squsize) for i in imgs ]
        imgs = np.array(imgs)
    elif len(imgs.shape)==3:
        if imgs.shape[-1]==3:# single RGB image
            ymx = imgs.shape[0]
            xmx = imgs.shape[1]
            bigger = ymx if ymx >= xmx else xmx
            squsize = (bigger, bigger)
            imgs = cv2.resize(src=imgs, dsize=squsize)
            imgs = np.array(imgs)
        else:# multiple grayscale images
            shapes = [ i.shape for i in imgs ]
            ymx = max([ i[0] for i in shapes ])
            xmx = max([ i[1] for i in shapes ])
            bigger = ymx if ymx >= xmx else xmx
            squsize = (bigger, bigger)
            imgs = [ cv2.resize(src=i, dsize=squsize) for i in imgs ]
            imgs = np.array(imgs)
    else:# single grayscale image
        ymx = imgs.shape[0]
        xmx = imgs.shape[1]
        bigger = ymx if ymx >= xmx else xmx
        squsize = (bigger, bigger)
        imgs = cv2.resize(src=imgs, dsize=squsize)
        imgs = np.array(imgs)
    return imgs


### Fanshape (FROM HERE) ###
def to_fan (imgs, angle=None, zero_offset=None, pix_per_mm=None, num_vectors=None, magnify=1, reserve=1800, show_progress=False ):
    if len(imgs.shape)==4:# multiple RGB images
        if show_progress:
            imgs = [ to_fan_2d(i, angle, zero_offset, pix_per_mm, num_vectors, magnify, reserve) for i in tqdm(imgs, desc='Fanshape')]
        else:
            imgs = [ to_fan_2d(i, angle, zero_offset, pix_per_mm, num_vectors, magnify, reserve) for i in imgs]
    elif len(imgs.shape)==3:
        if imgs.shape[-1]==3:# single RGB image
            imgs = to_fan_2d(imgs, angle, zero_offset, pix_per_mm, num_vectors, magnify, reserve)
        else:# multiple grayscale images
            if show_progress:
                imgs = [ to_fan_2d(i, angle, zero_offset, pix_per_mm, num_vectors, magnify, reserve) for i in tqdm(imgs, desc='Fanshape')]
            else:
                imgs = [ to_fan_2d(i, angle, zero_offset, pix_per_mm, num_vectors, magnify, reserve) for i in imgs]
    else:# single grayscale image
        imgs = to_fan_2d(imgs, angle, zero_offset, pix_per_mm, num_vectors, magnify, reserve)
    return np.array(imgs)

def to_fan_2d (img, angle=None, zero_offset=None, pix_per_mm=None, num_vectors=None, magnify=1, reserve=1800):

    use_genpar = any([ i is None for i in [angle, zero_offset, pix_per_mm, num_vectors] ])
    if use_genpar:
        print('WARNING: Not all the necessary information are provided. General parameters are used instead.')
        img = cv2.resize(img, (500,500))
        angle = 0.0031
        zero_offset = 150
        pix_per_mm = 2
        num_vectors = img.shape[0]

    pix_per_mm = pix_per_mm//magnify

    img = np.rot90(img, 3)
    dimnum = len(img.shape)
    if dimnum==2:
        grayscale = True
    elif dimnum==3 and img.shape[-1]==3:
        grayscale = False
    else:
        raise ValueError('Dimensions are not 2. And it does not look like a RGB format, either.')

    if grayscale:
        output_shape = (int(reserve // pix_per_mm), int( (reserve*0.80) // pix_per_mm))
    else:
        output_shape = (int(reserve // pix_per_mm), int( (reserve*0.80) // pix_per_mm), 3)
    origin = (int(output_shape[0] // 2), 0)

    img = ndimage.geometric_transform(img,
            mapping=ult_cart2pol,
            output_shape=output_shape,
            order=2,
            cval=255,
            extra_keywords={
                'origin': origin,
                'num_of_vectors': num_vectors,
                'angle': angle,
                'zero_offset': zero_offset,
                'pix_per_mm': pix_per_mm,
                'grayscale': grayscale})
    img = trim_picture(img)
    img = np.rot90(img, 1)
    return img

def ult_cart2pol(output_coordinates, origin, num_of_vectors, angle, zero_offset, pix_per_mm, grayscale):
    def cart2pol(x, y):
        r = math.sqrt(x**2 + y**2)
        th = math.atan2(y, x)
        return r, th
    (r, th) = cart2pol(output_coordinates[0] - origin[0],
                       output_coordinates[1] - origin[1])
    r *= pix_per_mm
    cl = num_of_vectors // 2
    if grayscale:
        res = cl - ((th - np.pi / 2) / angle), r - zero_offset
    else:
        res = cl - ((th - np.pi / 2) / angle), r - zero_offset, output_coordinates[2]
    return res

def trim_picture(img):
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
### Fanshape (UNTIL HERE) ###


def _temp_video ( imgs, fps, outpath='./video.avi' ):
    fps = int(fps)
    imgs = [ i for i in imgs ]
    if len(imgs)==1:
        print('WARNING: No video is produced because only an single image is provided.')
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

def _temp_audio (inpath, fps, outpath='', trim_beginning=0):
    if outpath=='':
        inpath = Path(inpath)
        outpath = str(inpath.parent) + '/' + inpath.stem + '_temp' + inpath.suffix
    cmd = ['sox', inpath, outpath, 'trim', str(trim_beginning), 'rate', '16000']
    subprocess.call(cmd)
    return None

def sync_audio_video ( invideo, inaudio, outvideo ):
    cmd = ['ffmpeg', '-i', invideo, '-i', inaudio, '-c:v', 'copy', '-c:a', 'aac', outvideo]
    subprocess.call(cmd)
    return None

def average_imgs (imgs):
    def same_shape (imglist):
        shps = [ i.shape for i in imglist ]
        res = len(set(shps))==1
        return res
    if isinstance(imgs, list):
        if not same_shape(imgs):
            shps = [ i.shape for i in imgs ]
            dim0 = max([ i[0] for i in shps ])
            dim1 = max([ i[1] for i in shps ])
            imgs = [ cv2.resize(src=i, dsize=(dim0, dim1)) for i in imgs ]
            warnings.warn('Input images have different dimensions: they are resized, so they share the same dimensions.')
        imgs = np.stack(imgs)
    dimnum = len(imgs.shape)
    if dimnum==4:# multiple RGB images
        raise ValueError('RGB images are not available for averaging images.')
    elif dimnum==3:
        if imgs.shape[-1]==3:# single RGB image
            raise ValueError('RGB images are not available for averaging images.')
        else:# multiple grayscale images
            mean_img = np.mean(imgs, axis=0)
            mean_img = mean_img.astype(int)
    else:# single grayscale image
        warnings.warn('Single image is passed: Nothing is done, returning the input image untouched.')
        mean_img = imgs
    return mean_img

def filter_imgs (imgs, frame=None, time=None, fps=None):
    flg = sum([ i is None for i in [frame, time] ])
    if flg==2 or flg==0:
        raise ValueError('Only one of frame or time should be provided.')
    if not time is None:
        if fps is None:
            raise ValueError("Provide fps when filtering by time.")
        else:
            frames_in_seconds = np.arange(1,len(imgs)+1)/fps
            try:
                frame = np.array([abs(frames_in_seconds-i).argmin() for i in time])
            except TypeError:
                frame = np.array(abs(frames_in_seconds-time).argmin())
    imgs = imgs[frame]
    return imgs

def ymax (img, ignore_edges=True):
    """
    Find the brightest pixel that is the highest along y-axis.
    Currently the function assumes a gray-scaled 2-dimensional image.

    Parameters
    ----------
    img : numpy.ndarray
        Input image you want to find the higheset y position in. It
        must be 2-dimensional (len(img.shape)==2) for the current
        version.
    ignore_edges : {True, False}
        If True, background pixels are ignored. This option should be
        kept True for a fan-shaped image, which has background pixels
        (usually white (255)) in the four corners. This option should
        be False for a normal image such as a raw (rectangle)
        ultrasound image.

    Returns
    -------
    pos_and_img : dict
        It has two keys, 'pos' and 'img'. 'pos' is the position of the
        brightest pixel highest along y-axis. 'img' is an RGB scale
        image, which is the same as the input image (img) but with a
        red horizontal line at the y-position of the brightest &
        highest pixel.

    """
    if len(img.shape)!=2:
        raise ValueError('''Dimension of the input image must be 2.
        It is now {}. If the image is in RGB scale, convert it into
        grayscale first. If the input is already in grayscale but has
        multiple frames, parallelize a function with
        pyult.image.parallelize or simply use the multiprocessing
        module.'''.format(len(img.shape)))
    if ignore_edges:
        edges = dict()
        for i in range(img.shape[1]):
            if img[0,i] in [0,255]:
                for ind,j in enumerate(img[:,i]):
                    if (int(img[0,i])-int(j))!=0:
                        img[:ind,i] = 0
                        edges[i] = list(range(ind))
                        break
        for i in range(img.shape[1]):
            if img[-1,i] in [0,255]:
                for ind,j in enumerate(img[::-1,i]):
                    if (int(img[-1,i])-int(j))!=0:
                        img[(len(img[:,i])-ind):,i] = 0
                        edges[i] = edges.get(i,[]) + list(range(len(img[:,i])-ind, len(img[:,i])))
                        edges[i] = sorted(list(set(edges[i])))
                        break
    ftv = get_fitted_values(img)
    rng = list(range(img.shape[1]))[int(abs(img.shape[1]*1/4)):int(abs(img.shape[1]*3/4))]
    ind = np.array([ i for i,j in enumerate(ftv['index']) if j in rng ])
    ftv = { i:j[ind] for i,j in ftv.items() }
    ymaxpos = min(ftv['fitted_values'])
    if ignore_edges:
        for i,j in edges.items():
            for k in j:
                img[k,i] = 255
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in range(img.shape[1]):
        img[ymaxpos, i, :] = (0,0,255)
    pos_and_img = {'pos':ymaxpos, 'img':img}
    return pos_and_img

def parallelize (imgs, func, cores=2):
    """
    Execute a function for each image in parallel.
    This function assumes 'imgs' is multi-frames of grayscale
    images for the current version.

    Parameters
    ----------
    imgs : numpy.ndarray (3d) or list of numpy.ndarray(2d)
        Input multi-frame images. Each image is assumed to be in
        grayscale in the current version.
    func : function
        Function to be applied to each (2-dimensional grayscale)
        image.
    cores : int
        How many cores to use.

    Return
    ------
    imgs : list of numpy.ndarray (2d)
        Images on which 'func' has been carried out.

    """
    if type(imgs) is np.ndarray:
        if len(imgs.shape)!=3:
            err = True
        else:
            err = False
    elif type(imgs) is list:
        check1 = all([ type(i) is np.ndarray for i in imgs ])
        if check1:
            check2 = all([ (i.shape)==2 for i in imgs ])
            if check2:
                err = False
            else:
                err = True
        else:
            err = True
    else:
        err = True
    if err:
        raise ValueError('''Input must be 3-dimensional
        numpy.ndarray or a list of 2-dimensional
        numpy.ndarray.''')
    with Pool(cores) as p:
        imgs = p.map(func, imgs)
    return imgs

def highest_xy (img, ignore_edges=True):
    """
    Find the brightest pixel that is the highest along y-axis.
    This function assumes the input as an RGB image.

    Parameters
    ----------
    img : numpy.ndarray
        Input image you want to find the higheset y position in. It
        must be 3-dimensional (len(img.shape)==3) and the length of
        the third axis (len(img.shape[2])==3) must be 3.
    ignore_edges : {True, False}
        If True, background pixels are ignored. This option should be
        kept True for a fan-shaped image, which has background pixels
        (usually white ([255,255,255])) in the four corners. This option should
        be False for a normal image such as a raw (rectangle)
        ultrasound image.

    Returns
    -------
    pos_and_img : dict
        It has two keys, 'ypos', 'xpos' and 'img'. 'ypos' is the position of the
        brightest pixel highest along y-axis. 'xpos' is its corresponding horizontal position. 'img' is an RGB scale image. The current version of this function assumes that the input is a spline-fitted image, where only the fitted spline is red. Accordingly, this function removes the red pixels from the spline curve. At the same time, blue horizontal and vertical lines are introduced, where the highest tongue contour point is located.
    """
    if ignore_edges:
        edges = dict()
        for i in range(img.shape[1]):
            if tuple(img[0,i]) in [(0,0,0),(255,255,255)]:
                for ind,j in enumerate(img[:,i]):
                    if tuple(img[0,i])!=tuple(j):
                        img[:ind,i] = (0,0,0)
                        edges[i] = list(range(ind))
                        break
        for i in range(img.shape[1]):
            if tuple(img[-1,i]) in [(0,0,0),(255,255,255)]:
                for ind,j in enumerate(img[::-1,i]):
                    if tuple(img[-1,i])!=tuple(j):
                        img[(len(img[:,i])-ind):,i] = (0,0,0)
                        edges[i] = edges.get(i,[]) + list(range(len(img[:,i])-ind, len(img[:,i])))
                        edges[i] = sorted(list(set(edges[i])))
                        break
    ymaxpos = None
    for i in range(img.shape[0]):
        mar = int(img.shape[1]//2.125)
        is_red = [tuple(j) == (0,0,255) for j in img[i,mar:-mar,:]]
        if any(is_red):
            ymaxpos = i
            xmaxpos = np.where(is_red)[0]
            xmaxpos = mar + xmaxpos[len(xmaxpos)//2]
            break
    if ymaxpos is None:
        raise ValueError('There is no red pixel in the picture')
    if ignore_edges:
        for i,j in edges.items():
            for k in j:
                img[k,i] = (255,255,255)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            crgb = img[i,j,:]
            if not (crgb==crgb[0]).all():
                pxls = []
                for k in range(-1,2):
                    for l in range(-1,2):
                        trgb = img[i+k, j+l, :]
                        if (trgb==trgb[0]).all():
                            pxls.append(tuple(trgb))
                rgb = []
                for k in range(0,3):
                    rgb.append(np.mean([ i[k] for i in pxls ], dtype=int))
                img[i,j,:] = tuple([np.mean(rgb,dtype=int)]*3)
    for i in range(img.shape[0]):
        img[i, xmaxpos, :] = (255,0,0)
    for i in range(img.shape[1]):
        img[ymaxpos, i, :] = (255,0,0)
    pos_and_img = {'ypos':ymaxpos, 'xpos':xmaxpos, 'img':img}
    return pos_and_img

