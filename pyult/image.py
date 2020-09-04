import numpy as np
import cv2
import pyult.ncspline as spl

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
    imgs = [ __crop2d(i, crop_points) for i in imgs ]
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
    imgs = [ __crop2d(i, crop_points) for i in imgs ]
    imgs = np.array(imgs)
    return imgs

def flip (imgs, direct):
    if direct=='x':
        direct = 1
    elif direct=='y':
        direct = 0
    elif direct=='xy':
        direct = (0,1)
    else:
        raise ValueError('The second argument "direct" should be "x", "y", or "xy".')
    imgs = [ np.flip(i, direct) for i in imgs ]
    imgs = np.array(imgs)
    return imgs

def reduce_y (imgs, every_nth):
    imgs = [ i[::every_nth, :] for i in imgs ]
    imgs = np.array(imgs)
    return imgs

### Functions for fitting spline curves (FROM HERE) ###
def fit_spline (imgs, return_fitted_values=False):
    imgs = [ fit_spline_2d(i, return_fitted_values) for i in imgs ]
    if return_fitted_values:
        images = [ i['image'] for i in imgs ]
        ftvs = [ i['fitted_values'] for i in imgs ]
        ret = {'images':np.array(images), 'fitted_values':ftvs}
    else:
        ret = np.array(imgs)
    return ret

def fit_spline_2d (img, return_fitted_values=False):
    ftv = get_fitted_values(img)
    yyy = ftv['fitted_values']
    xxx = ftv['index']
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for xpos,ypos in zip(xxx, yyy):
        if not ypos is None:
            img[ypos-2:ypos+2, xpos-2:xpos+2, :] = (0,0,255)
    if return_fitted_values:
        ret = {'image':img, 'fitted_values':ftv}
    else:
        ret = img
    return ret

def get_fitted_values (img):
    pks = [ spl.fit_ncspline(img[:,i]) for i in range(img.shape[1])]
    pks = [ peaks(i) for i in pks ]
    pks = { i:j for i,j in enumerate(pks) }
    aaa = len(pks)//4
    bbb = np.arange(aaa, len(pks)-aaa)
    selpks = { i:j for i,j in pks.items() if i in bbb }
    aaa = max([ i['peak_ratio'] for i in selpks.values() ])
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
    y = np.array(list(xy1.values()))
    x = np.array(list(xy1.keys()))
    nxmin = np.array(list(xy1.keys())).astype(int).min()
    nxmax = np.array(list(xy1.keys())).astype(int).max()
    nx = np.arange(nxmin, nxmax+1)
    ftv = spl.fit_ncspline(y=y, x=x, knots=10, pred_x=nx)
    ftv = ftv.round().astype(int)
    ftv_dict = {'index':nx, 'fitted_values':ftv}
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
    peak_dict = { i:j for i,j in zip(peak_poses, peak_vals) if i <= len(f0)*3/4 }
    max_vals = np.sort(np.array(list(peak_dict.values())))[::-1][:howmany]
    peak_dict = { i:j for i,j in peak_dict.items() if j in max_vals }
    peak_poses = np.array(list(peak_dict.keys()))
    peak_vals = np.array(list(peak_dict.values()))
    aaa = np.sort(peak_vals)[::-1][:2]
    peak_ratio = aaa[0]/aaa[1]
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





