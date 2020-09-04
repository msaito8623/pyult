import numpy as np
import pandas as pd

def imgs_to_df (imgs, fps=None):
    imgs = [ img_to_df(img=i, frame_id=frame) for frame,i in enumerate(imgs) ]
    df = pd.concat(imgs, ignore_index=True)
    if not fps is None:
        df['time'] = df['frame'] * (1/fps)
    return df

def img_to_df (img, frame_id=None):
    ult  = img.flatten()
    xlen = img.shape[1]
    ylen = img.shape[0]
    xxx = np.array(list(range(xlen))*ylen)
    yyy = np.repeat(np.arange(ylen), xlen)
    df = pd.DataFrame({'brightness':ult, 'x':xxx, 'y':yyy})
    if not frame_id is None:
        df['frame'] = frame_id
    return df

def integrate_segments ( df, dfphones, dfwords, rmvnoise=True ):
    dfphones = alignment_df(dfphones, dfwords)
    df['segment'] = ''
    df['word'] = ''
    for i in dfphones.index:
        cst = dfphones.loc[i,'start']
        ced = dfphones.loc[i,'end']
        csg = dfphones.loc[i,'segment']
        cwd = dfphones.loc[i,'word']
        df.loc[ (df.time>cst) & (df.time<ced), 'segment'] = csg
        df.loc[ (df.time>cst) & (df.time<ced), 'word'] = cwd
    if rmvnoise:
        df = rmv_noise(df)
    return df

def alignment_df (dfphones, dfwords):
    dfphones['start'] = dfphones['end'].shift(fill_value=0)
    dfwords['start'] = dfwords['end'].shift(fill_value=0)
    def word_now (value, wrds):
        res = [ k for i,j,k in zip(wrds.start, wrds.end, wrds.word) if (value>i) and (value<=j) ]
        if len(res)!=1:
            raise ValueError('The provided value corresponds to more than one word.')
        return res[0]
    dfphones['word'] = [ word_now(i, dfwords) for i in dfphones['end'] ]
    return dfphones

def rmv_noise (df):
    noise = ['_p:_','<P>','_NOISE_','<NOISE>']
    colnames=['segment','word']
    for i in colnames:
        pos1 = df[i].isin(noise)
        pos2 = df[i].isna() 
        pos3 = pd.Series([ j=='' for j in df[i] ])
        df = df.loc[~((pos1|pos2)|pos3),:]
    return df

def integrate_splines ( df, splvals ):
    splvals = { i:j for i,j in enumerate(splvals) if i in set(df.frame) }
    def __todf (frame, content):
        dat = pd.DataFrame(content)
        dat['frame'] = frame
        dat['x'] = dat.pop('index')
        dat['y_spline'] = dat.pop('fitted_values')
        return dat
    splvals = [ __todf(i,j) for i,j in splvals.items() ]
    splvals = pd.concat(splvals, ignore_index=True)
    df = pd.merge(df, splvals, on=['frame','x'], how='left')
    return df

