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
        csg = dfphones.loc[i,'segments']
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
        res = [ k for i,j,k in zip(wrds.start, wrds.end, wrds.words) if (value>i) and (value<=j) ]
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

    labels = [ i.replace('name = "','') for i in lines if 'name = "' in i ]
    labels = [ i.replace('"','').strip() for i in labels ]
    label1 = labels[0]
    label2 = labels[1]
    if len(labels)>2:
        label3 = labels[2]
        keys = [label1, label2, label3, 'xmin', 'xmax', 'text']
    else:
        keys = [label1, label2, 'xmin', 'xmax', 'text']

    pos = []
    for i in keys:
        pos = pos + [ j for j,k in enumerate(lines) if i in k ]
    pos = sorted(list(set(pos)))
    lines = lines[pos]

    if len(labels)>2:
        pos = [ i for i,j in enumerate(lines) if (label1 in j) or (label2 in j) or (label3 in j)]
    else:
        pos = [ i for i,j in enumerate(lines) if (label1 in j) or (label2 in j) ]

    interval1 = lines[pos[0]:pos[1]]
    if len(pos)>2:
        interval2 = lines[pos[1]:pos[2]]
    else:
        interval2 = lines[pos[1]:]
    interval1 = [ i.strip() for i in interval1 ]
    interval2 = [ i.strip() for i in interval2 ]
    interval1 = [ i.split(' = ')[1] for i in interval1 ]
    interval2 = [ i.split(' = ')[1] for i in interval2 ]
    interval1 = _temp(interval1)
    interval2 = _temp(interval2)
    interval1 = [ i + ' 000 ' + j for i,j in zip(interval1.xmax, interval1.text) ]
    interval2 = [ i + ' 000 ' + j for i,j in zip(interval2.xmax, interval2.text) ]
    interval1 = ['#'] + interval1
    interval2 = ['#'] + interval2

    def __todf (line, typ):
        line = line.split(' ')
        if len(line)==3:
            ret = list(np.array(line)[[0,2]])
            ret = pd.DataFrame([ret], columns=['end',typ])
            ret['end'] = ret['end'].astype(float)
        else:
            ret = None
        return ret
    interval1 = pd.concat([ __todf(i, label1) for i in interval1 ], ignore_index=True)
    interval2 = pd.concat([ __todf(i, label2) for i in interval2], ignore_index=True)
    return {label1:interval1, label2:interval2}

def textgrid_to_df ( textgridlist ):
    aln = textgrid_to_alignfiles(textgridlist)
    assert len(aln)==2
    keys = list(aln.keys())
    values = list(aln.values())
    lens = [ len(i) for i in values ]
    short_long = lens[0] < lens[1]
    if short_long:
        df_short  = values[0]
        df_long   = values[1]
        key_short = keys[0]
        key_long  = keys[1]
    else:
        df_short  = values[1]
        df_long   = values[0]
        key_short = keys[1]
        key_long  = keys[0]
    df_short['ID'+key_short] = [ '{}-{}'.format(i,j) for i,j in enumerate(df_short[key_short]) ]
    df_long['ID'+key_long] = [ '{}-{}'.format(i,j) for i,j in enumerate(df_long[key_long]) ]
    df_long['start'] = df_long.end.shift(fill_value=0)
    df_short['start'] = df_short.end.shift(fill_value=0)
    df_short = df_short.reset_index(drop=True)
    for i in range(len(df_short)):
        cstart = df_short.loc[i,'start']
        cend   = df_short.loc[i,'end']
        cid    = df_short.loc[i,'ID'+key_short]
        if cstart==0:
            pos1 = df_long.end>=cstart
        else:
            pos1 = df_long.end>cstart
        pos2 = df_long.end<=cend
        pos = pos1 & pos2
        df_long.loc[pos, 'ID'+key_short] = cid
    df_long  = df_long.rename(columns={'end':'end_{}'.format(key_long)})
    df_short = df_short.rename(columns={'end':'end_{}'.format(key_short)})
    df_long  = df_long.rename(columns={'start':'start_{}'.format(key_long)})
    df_short = df_short.rename(columns={'start':'start_{}'.format(key_short)})
    df_merged = pd.merge(df_long, df_short, on='ID'+key_short, how='left')
    kept_cols = ['ID'+key_long, key_long, 'start_'+key_long, 'end_'+key_long, 'ID'+key_short, key_short, 'start_'+key_short, 'end_'+key_short]
    df_merged = df_merged.loc[:,kept_cols]
    return df_merged


