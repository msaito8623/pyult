import numpy as np
import pandas as pd
from pathlib import Path

def imgs_to_df (imgs, fps=None):
    imgs = [ img_to_df(img=i, frame_id=frame) for frame,i in enumerate(imgs) ]
    df = pd.concat(imgs, ignore_index=True)
    if not fps is None:
        df['time'] = df['frame'] * (1/fps)
    return df

def img_to_df (img, frame_id=None, fps=None):
    ult  = img.flatten()
    xlen = img.shape[1]
    ylen = img.shape[0]
    xxx = np.array(list(range(xlen))*ylen)
    yyy = np.repeat(np.arange(ylen), xlen)
    df = pd.DataFrame({'brightness':ult, 'x':xxx, 'y':yyy})
    if not frame_id is None:
        df['frame'] = frame_id
        if not fps is None:
            df['time'] = df['frame'] * (1/fps)
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
    colnames = ['segment','word']
    exist_col_pos = [ i in df.columns for i in colnames ]
    colnames = np.array(colnames)[exist_col_pos]
    for i in colnames:
        pos1 = df[i].isin(noise)
        pos2 = df[i].isna() 
        pos3 = pd.Series([ j=='' for j in df[i] ])
        df = df.loc[~((pos1|pos2)|pos3),:]
    return df

def integrate_splines ( df, splvals ):
    if type(splvals) is list:
        splvals = { i:j for i,j in enumerate(splvals) if i in set(df.frame) }
    elif type(splvals) is dict:
        assert len(set(df.frame))==1
        splvals = {df.frame.iloc[0]: splvals}
    else:
        raise ValueError('fitted_values is invalid.')
    def __todf (frame, content):
        if content is None:
            dat = None
        else:
            dat = pd.DataFrame(content)
            dat['frame'] = frame
            dat['x'] = dat.pop('index')
            dat['y_spline'] = dat.pop('fitted_values')
        return dat
    splvals = [ __todf(i,j) for i,j in splvals.items() ]
    splvals = pd.concat(splvals, ignore_index=True)
    df = pd.merge(df, splvals, on=['frame','x'], how='left')
    return df

def textgrid_to_alignfiles ( textgridlist ):
    def _temp ( xxx ):
        xxx = np.array(xxx)
        xxx = xxx[3:]
        if len(xxx)%3!=0:
            raise ValueError('Something is wrong in the format of the input textgrid.')
        xmins = np.arange(0, len(xxx), 3)
        xmaxs = np.arange(1, len(xxx), 3)
        texts = np.arange(2, len(xxx), 3)
        xmins = xxx[xmins]
        xmaxs = xxx[xmaxs]
        texts = xxx[texts]
        xmins = [ '{:8.6f}'.format(float(i)) for i in xmins ]
        xmaxs = [ '{:8.6f}'.format(float(i)) for i in xmaxs ]
        texts = [ i.strip('\"') for i in texts ]
        df = pd.DataFrame({'xmin':xmins, 'xmax':xmaxs, 'text':texts})
        return df

    lines = np.array(textgridlist)
    keys = ['segments', 'words', 'xmin', 'xmax', 'text']
    pos = []
    for i in keys:
        pos = pos + [ j for j,k in enumerate(lines) if i in k ]
    pos = sorted(list(set(pos)))
    lines = lines[pos]

    pos = [ i for i,j in enumerate(lines) if ('segments' in j) or ('words' in j) ]
    if len(pos)!=2 :
        print('WARNING: Multiple lines matched "segments" and "words" in a textgrid file.')

    segments = lines[pos[0]:pos[1]]
    words = lines[pos[1]:]
    segments = [ i.strip() for i in segments ]
    words = [ i.strip() for i in words ]
    segments = [ i.split(' = ')[1] for i in segments ]
    words = [ i.split(' = ')[1] for i in words ]
    segments = _temp(segments)
    words = _temp(words)
    segments = [ i + ' 000 ' + j for i,j in zip(segments.xmax, segments.text) ]
    words = [ i + ' 000 ' + j for i,j in zip(words.xmax, words.text) ]
    segments = ['#'] + segments
    words = ['#'] + words

    def __todf (line, typ):
        line = line.split(' ')
        if len(line)==3:
            ret = list(np.array(line)[[0,2]])
            ret = pd.DataFrame([ret], columns=['end',typ])
            ret['end'] = ret['end'].astype(float)
        else:
            ret = None
        return ret
    segments = pd.concat([ __todf(i, 'segment') for i in segments ], ignore_index=True)
    words = pd.concat([ __todf(i, 'word') for i in words], ignore_index=True)
    return {'phones':segments, 'words':words}

