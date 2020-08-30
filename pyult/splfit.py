"""
pyult.splfit
----------

*pyult.splfit* provides functions to fit a spline curve on an image.

"""
import cv2
import multiprocessing as mp
import numpy as np
import pyper
from . import ncspline as spl

def fit_ncspline (y, x=None, knots=0):
    if x is None:
        x = np.arange(len(y))
    if knots==0:
        knots = len(y)/5
    mdl = spl.get_natural_cubic_spline_model(x, y, minval=min(x), maxval=max(x), n_knots=knots)
    est = mdl.predict(x)
    return est

def fitted_values ( vect, pred=None, knots=30, png_out=False, outpath=None, lwd=1, col=('black','red'), xlab='', ylab=''):
    r = pyper.R()
    r('ready_mgcv = require(mgcv)')
    ready_mgcv = r('ready_mgcv').split(' ')[1].strip()


    if all([ready_mgcv=='TRUE']):
        vect = np.array(vect, dtype=np.float)
        r.assign('vect', vect)
        if pred is None:
            r('xxx = seq_along(vect)')
        else:
            pred = np.array(pred, dtype=np.float)
            r.assign('xxx', pred)
        r('mdl = gam(vect ~ s(xxx, k={}))'.format(knots))
        r('ndat = data.frame("xxx"=xxx)')
        r('ftv = as.vector(predict(mdl, newdata=ndat))')
        ftv = r.get('ftv')
        #
        if png_out:
            if outpath is None:
                raise ValueError('outpath is necessary when png_out is True.')
            if isinstance(lwd, (int,float)):
                lwd = (lwd,lwd)
            if isinstance(lwd, str):
                col = (col,col)
            r('rng = range(c(vect, mdl$fitted.values))')
            r('png("{}", width=500, height=500)'.format(outpath))
            r('plot(vect, ylim=rng, type="l", lwd={}, col="{}", xlab="{}", ylab="{}")'.format(lwd[0],col[0],xlab,ylab))
            r('par(new=T)')
            r('plot(mdl$fitted.values, ylim=rng, type="l", lwd={}, col="{}", xlab="{}", ylab="{}")'.format(lwd[1],col[1],xlab,ylab))
            r('dev.off()')
    else:
        raise ImportError('You need to install mgcv in R first.')
    return ftv

def deriv_discrete ( vect ):
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

def peaks ( vect, howmany=3 ):
    def __cross_zero ( nums ):
        return (nums[0]>0 and nums[1]<0) or (nums[0]<0 and nums[1]>0)
    f0 = vect
    d1 = deriv_discrete(f0)
    d2 = deriv_discrete(d1)
    d1_0 = np.where(d1==0)[0]
    if len(d1_0)==0:
        aaa = [ __cross_zero(d1[i:(i+2)]) for i in range(len(d1)-1) ]
        aaa = np.array(list(aaa) + [False])
        d1_0 = np.where(aaa)[0]
    try:
        aaa = d1_0 + 1
        bbb = np.concatenate([d1_0[1:],np.arange(1)])
        rmpos = np.where(aaa==bbb)[0]+1
        ccc = np.delete(d1_0, rmpos)
    except TypeError:
        pass
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

def _fitted_values ( vect ):
    return fitted_values(vect)

def fit_spline ( img, cores=1, inplace=False ):
    if cores != 1:
        pool = mp.Pool(cores)
        pks = pool.map(_fitted_values, [img[:,i] for i in range(img.shape[1])])
    else:
        pks = [ fitted_values(img[:,i]) for i in range(img.shape[1])]
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

    xy = { i:j['peak_poses'] for i,j in pks.items() }
    ftv = fitted_values(vect=list(xy.values()), pred=list(xy.keys()), knots=10)
    ftv = ftv.round().astype(int)
    ftv_dict = {'index':np.array(list(xy.keys())), 'fitted_values':ftv}
    if inplace:
        spline_values = ftv_dict
        ftv_dict = None
    return ftv_dict

def fit_spline_img ( img, ftv=None, cores=1 ):
    if ftv is None:
        ftv = fit_spline(img=img, cores=cores)
    yyy = ftv['fitted_values']
    xxx = ftv['index']
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for xpos,ypos in zip(xxx, yyy):
        if not ypos is None:
            img[ypos-2:ypos+2, xpos-2:xpos+2, :] = (0,0,255)
    return img

