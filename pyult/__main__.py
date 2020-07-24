import pyult
upc = pyult.UltPicture()
ufl = pyult.Files()
import argparse
parser = argparse.ArgumentParser()
parser.parse_args()
import os
import pandas as pd
import pdb
import subprocess
import shutil
import copy
from tqdm import tqdm


### Generic ###
def check_yes_or_no ( inpt ):
    ok = inpt.lower() in ['yes', 'no', 'y', 'n']
    if not ok:
        print("Please type 'yes' or 'no'")
    return ok 

def yesno_to_bool ( yn ):
    yeses  = ['yes', 'y']
    noes = ['no', 'n']
    opts = yeses + noes
    if yn in yeses:
        res = True
    elif yn in noes:
        res = False
    else:
        err = ', '.join(opts)
        err = 'Following inputs are acceptable: {}'.format(err)
        raise ValueError(err)
    return res

def check_paths_alright ( obj, directory=None ):
    if not directory is None:
        obj.set_paths(directory)
    if not obj.is_all_set(exclude_empty_list=True):
        raise FileNotFoundError('Problem detected in the files in the directory specified.')
    return None

def ask_cores () :
    print('\n##########')
    print('- How many cores do you want to use?')
    ok = False
    while not ok:
        cores = input()
        try:
            cores = int(cores)
            ok = True
        except ValueError:
            print('cores must be an integer.')
    return cores

def determine_cores ( whichimgtype ):
    aaa = set(whichimgtype)
    bbb = set(['all','fan'])
    ccc = aaa.intersection(bbb)
    if len(ccc)>0:
        cores = ask_cores()
    else:
        cores = 1
    return cores

def create_dir ( obj, dirname ):
    newdir = obj.wdir + '/' + dirname
    os.makedirs(newdir, exist_ok=True)
    return newdir
### Generic ###

### Initiation ###
def ask_target_dir (obj):
    print('\n##########')
    print('- Give me a file path or a directory path which has all the necessary files.')
    ok = False
    while not ok:
        tdir = input()
        if obj.exist(tdir):
            obj.set_paths(tdir)
            if obj.is_all_set(exclude_empty_list=True):
                ok = True
            else:
                print('- The directory specified does not have all necessary files ready. Please try again.')
        else:
            print('- The path specified does not exist. Please try again.')
    return obj

def ask_whattodo ():
    print('\n##########')
    print('- What do you want from the file(s)?')
    print('--- Pictures             => pic')
    print('--- Videos               => vid')
    print('--- Dataframes           => df')
    print('--- TextGrids            => tg')
    print('--- If you want them all => all')
    print('--- You can specify them together with a space, e.g. pic vid')
    ok_inputs = [ 'pic', 'vid', 'df', 'tg', 'all' ]
    ok_while = False
    while not ok_while:
        whattodo = input()
        whattodo = whattodo.split(' ')
        whattodo = [ i for i in whattodo if i != '' ]
        oks = [ i for i in whattodo if     i in ok_inputs ]
        ngs = [ i for i in whattodo if not i in ok_inputs ]
        if len(oks)==0:
            print('- Please choose out of the followings:')
            print('--- {}'.format(', '.join(ok_inputs)))
        else:
            ok_while = True
    if len(ngs)>0:
        ngs = ', '.join(ngs)
        print('')
        print('- WARNING: Following inputs are ignored.')
        print('--- {}'.format(ngs))
    return oks

def whattodo_to_flags ( whattodo ):
    picT = 'pic' in whattodo
    vidT = 'vid' in whattodo
    dfT = 'df' in whattodo
    tgT = 'tg' in whattodo
    if 'all' in whattodo: picT,vidT,dfT,tgT = [True]*4
    return (picT, vidT, dfT, tgT)
### Initiation ###

### TextGrid ###
def produce_textgrids ( obj ):
    alhome = ask_aligner_home()
    keeptmp = ask_keep_tmp_files()
    aln = pyult.Alignment()
    for i,j,k in zip(obj.paths['wav'],obj.paths['txt'],obj.paths['ustxt']):
        obj.read(k,inplace=True)
        temp_a = obj.replace('.wav', '_temp.wav', i)
        aln.prepare_audio(i, temp_a, ff=obj.timeinsecsoffirstframe)
        obj.to_alignment_txt(j,temp_a)
        aln.run_aligner(temp_a, alhome)
        j = aln.replace('.wav', '.phoneswithQ', temp_a)
        k = aln.replace('.wav', '.words', temp_a)
        aln.read_align_file(j, inplace=True)
        aln.read_align_file(k, inplace=True)
        aln.format_align_files()
        aln.to_textgrid(temp_a)

        cdir = obj.wdir
        tmpdir = cdir + '/TMP'
        tmpfilename = obj.name(temp_a, drop_extension=True)
        nmlfilename = obj.name(i, drop_extension=True)

        kp_sfxs = ['.phoneswithQ', '.TextGrid', '.words']
        kp_files_fr = [ '{}/{}{}'.format(cdir,tmpfilename,i) for i in kp_sfxs ]
        kp_files_to = [ '{}/{}{}'.format(cdir,nmlfilename,i) for i in kp_sfxs ]
        for i,j in zip(kp_files_fr, kp_files_to):
            os.rename(i,j)

        ot_sfxs = ['.mfc', '.phonemic', '.phones', '.txt', '.wav']
        files_fr = [ '{}/{}{}'.format(cdir,tmpfilename,i) for i in ot_sfxs ]
        files_to = [ '{}/{}{}'.format(tmpdir,nmlfilename,i) for i in ot_sfxs ]
        if keeptmp=='yes':
            os.makedirs(tmpdir, exist_ok=True)
            for i,j in zip(files_fr, files_to):
                shutil.move(i,j)
        elif keeptmp=='no':
            for i in files_fr:
                os.remove(i)
        else:
            raise ValueError('yes or no is acceptable')
    return None
def ask_aligner_home ():
    print('\n##########')
    print('- Where is the home directory for Aligner? (E.g. /home/username/Aligner)')
    print('--- As default, it should contain AUTHORS, bin, BUGS, confs, dict, ...and so on.')
    ok = False
    while not ok:
        alhome = input()
        if ufl.exist(alhome):
            ok = True
        else:
            print('- The path specified does not exist. Please try again.')
    return alhome
def ask_keep_tmp_files ():
    print('\n##########')
    print('- Do you want to keep temporary files (yes or no)')
    print('--- E.g. Do you want to keep a wav file whose sampling rate is 16kHz? The produced TextGrid file is based on this audio file. You should open it with its corresponding TextGrid file if you want to align and display them in Praat.')
    ok = False
    while not ok:
        keeptmp = input()
        if check_yes_or_no(keeptmp):
            ok = True
    return keeptmp
### TextGrid ###

### Prepare images ###
def prepare_imgs ( obj, ind ):
    obj.read(obj.paths['ult'][ind], inplace=True)
    obj.read(obj.paths['ustxt'][ind], inplace=True)
    obj.vec_to_pics(inplace=True)
    return obj
###

### Cropping ###
def ask_crop_ornot ():
    print('\n##########')
    print('- Do you want to crop pictures? (yes or no)')
    ok = False
    while not ok:
        cropornot = input()
        if check_yes_or_no(cropornot):
            ok = True
    return cropornot

def ask_wheretocrop ():
    print('\n##########')
    print('- Where to crop?')
    print('--- Specify as a 4-element tuple, i.e. (xmin, xmax, ymin, ymax).')
    print('--- Please type None if you want to refer to the ends of x/y-axes.')
    print('--- E.g. "You want to cut off only the first 10 pixels along x-axis"')
    print('      ==> (10, None, None, None)')
    ok = False
    while not ok:
        crp = input()
        tgt = [ ' ', '(', ')' ]
        for i in tgt:
            crp = crp.replace(i, '')
        crp = crp.split(',')
        try:
            crp = tuple(map(lambda x: int(x) if x!='None' else None, crp))
            ok = True
        except ValueError:
            print('- Input must follow the following format: e.g. (10, None, 20, None)')
            print('- Please try again...')
    return crp
        
def where_to_crop ():
    cropornot = ask_crop_ornot()
    crp = yesno_to_bool(cropornot)
    if crp:
        crp = ask_wheretocrop()
    return crp

def crop_local ( obj, crp ):
    if crp:
        obj.crop(crp, inplace=True)
    return obj
### Cropping ###

### Reduce Resolution ###
def ask_reduce_resolution_ornot ():
    print('\n##########')
    print('- Do you want to reduce resolution of y-axis? (yes or no)')
    print('(Because ultasound pictures have much more information along y-axis than x-axis, it might not hurt to take every n-th pixel along y-axis to reduce data size.)')
    ok = False
    while not ok:
        resolreduc = input()
        if check_yes_or_no(resolreduc):
            ok = True
    return resolreduc

def ask_howmuchreduce ():
    print('\n##########')
    print('- How much to reduce?')
    print('--- E.g. 3')
    print('------> meaning you take every 3rd pixel along y-axis')
    print('------> leading to approximately 1/3 of the original data size.')
    ok = False
    while not ok:
        nth = input()
        try:
            nth = int(nth)
            ok = True
        except ValueError:
            print('- Only integers are acceptable. Please try again...')
    return nth
        
def is_resol_reduc_needed ():
    rsl = ask_reduce_resolution_ornot()
    rsl = yesno_to_bool(rsl)
    if rsl:
        rsl = ask_howmuchreduce()
    return rsl

def reduce_resolution_local ( obj, rsl ):
    if rsl:
        obj.reduce_resolution(every_y=rsl, inplace=True)
    return obj
### Reduce Resolution ###

### Change image shapes ###
def ask_which_imgtype ():
    print('\n##########')
    print('- What shape of pictures do you want?')
    print('--- Raw (rectangle) => raw')
    print('--- Square          => squ')
    print('--- Fan-shape       => fan')
    print('--- All             => all')
    print('--- You can specify them together with a space, e.g. raw squ')
    ok_inputs = [ 'raw', 'squ', 'fan', 'all' ]
    ok_while = False
    while not ok_while:
        whattodo = input()
        whattodo = whattodo.split(' ')
        whattodo = [ i for i in whattodo if i != '' ]
        oks = [ i for i in whattodo if     i in ok_inputs ]
        ngs = [ i for i in whattodo if not i in ok_inputs ]
        if len(oks)==0:
            print('- Please choose out of the followings:')
            print('--- {}'.format(', '.join(ok_inputs)))
        else:
            ok_while = True
    if len(ngs)>0:
        ngs = ', '.join(ngs)
        print('')
        print('- WARNING: Following inputs are ignored.')
        print('--- {}'.format(ngs))
    return oks

def change_img_shapes ( obj, imgtype, cores ):
    obj.raw = copy.deepcopy(obj.img)
    if imgtype is None:
        pass
    elif 'all' in imgtype:
        obj.squ = obj.to_square(inplace=False)
        obj.fan = obj.fanshape(inplace=False, numvectors=obj.img.shape[-1], magnify=4, cores=cores, progressbar=True)
    else:
        if 'squ' in imgtype:
            obj.squ = obj.to_square(inplace=False)
        if 'fan' in imgtype:
            obj.fan = obj.fanshape(inplace=False, numvectors=obj.img.shape[-1], magnify=4, cores=cores, progressbar=True)

    if hasattr(obj, 'splimg'):
        obj.splraw = copy.deepcopy(obj.splimg)
        if imgtype is None:
            pass
        elif 'all' in imgtype:
            obj.splsqu = [ obj.to_square(img=i, inplace=False, rgb=True) for i in obj.splimg ]
            obj.splfan = [ obj.fanshape(img=i, inplace=False, numvectors=obj.img.shape[-1], magnify=4, cores=cores, progressbar=True) for i in obj.splimg ]
        else:
            if 'squ' in imgtype:
                obj.splsqu = [ obj.to_square(img=i, inplace=False, rgb=True) for i in obj.splimg ]
            if 'fan' in imgtype:
                obj.splfan = [ obj.fanshape(img=i, inplace=False, numvectors=obj.img.shape[-1], magnify=4, cores=cores, progressbar=True) for i in obj.splimg ]
        delattr(obj, 'splimg')

    delattr(obj, 'img')
    return obj
### Change image shapes ###

### Flip ###
def ask_flip ( vertical=False ):
    if vertical:
        direction = 'vertically'
    else:
        direction = 'horizontally'
    print('\n##########')
    print('- Do you want to flip pictures {}? (yes or no)'.format(direction))
    opts = [ 'yes', 'no', 'y', 'n']
    ok = False
    while not ok:
        xyflip = input().lower()
        if xyflip in opts:
            ok = True
        else:
            print('- Please choose out of the followings:')
            print('--- {}'.format(', '.join(opts)))
    return xyflip

def ask_flip_to_flags ( xflip, yflip ):
    xxx = xflip in ['yes','y']
    yyy = yflip in ['yes','y']
    if xxx and yyy:
        flp = 'xy'
    elif xxx and not yyy:
        flp = 'x'
    elif not xxx and yyy:
        flp = 'y'
    else:
        flp = False
    return flp

def determine_flip_directions ():
    xflip = ask_flip(vertical=False)
    yflip = ask_flip(vertical=True)
    flp = ask_flip_to_flags(xflip, yflip)
    return flp

def flip_local ( obj, flip_directions ):
    if flip_directions:
        if hasattr(obj, 'raw'):
            obj.raw = obj.flip(flip_directions, img=obj.raw, inplace=False)
        if hasattr(obj, 'squ'):
            obj.squ = obj.flip(flip_directions, img=obj.squ, inplace=False)
        if hasattr(obj, 'fan'):
            obj.fan = obj.flip(flip_directions, img=obj.fan, inplace=False)
        if hasattr(obj, 'splraw'):
            obj.splraw = [ obj.flip(flip_directions, img=i, inplace=False, rgb=True) for i in obj.splraw ]
        if hasattr(obj, 'splsqu'):
            obj.splsqu = [ obj.flip(flip_directions, img=i, inplace=False, rgb=True) for i in obj.splsqu ]
        if hasattr(obj, 'splfan'):
            obj.splfan = [ obj.flip(flip_directions, img=i, inplace=False, rgb=True) for i in obj.splfan ]
    return obj
### Flip ###

### Spline Fitting ###
def ask_spline_fitting(boolean=False) :
    print('\n##########')
    print('- Do you want to fit spline curves? (yes or no)')
    ok = False
    while not ok:
        spl = input()
        if check_yes_or_no(spl):
            ok = True
    if boolean:
        spl = yesno_to_bool(spl)
    return spl

def spline_local (obj, spl, cores) :
    if spl:
        def temp (i,cores):
            import numpy as np
            try:
                print(i)
                print(type(i))
                print(i.shape)
                res = obj.fit_spline(img=i, cores=cores)
            except IndexError:
                np.savetxt('./error_matrix.txt', i)
                raise ValueError('ERROR!!! but I knew it.')
                res = None
            return res
        obj.splval = [ temp(i,cores) for i in tqdm(obj.img,desc='Spline') ]
        obj.splimg = [ obj.fit_spline_img(img=i,ftv=j,cores=cores) for i,j in zip(obj.img, obj.splval) ]
    return obj
### Spline Fitting ###

### Produce functions ###
def produce_df ( obj, path_index ):
    df_dir = create_dir(obj, 'Dataframes')
    udf = pyult.UltDf(obj)
    try:
        path_p = udf.paths['phoneswithQ'][path_index]
        path_w = udf.paths['words'][path_index]
        alfiles = True
    except IndexError:
        alfiles = False
        print('- WARNING: Alignment files (i.e. *.phoneswithQ and *.words) are not found, therefore segments/words information is not integrated into produced dataframes)')
    fname = udf.name(udf.paths['ult'][path_index], drop_extension=True)
    udf.df = udf.img_to_df(img=udf.raw, add_time=True, combine=False)
    udf.df = [ udf.integrate_spline_values(i,j) for i,j in zip(udf.df, udf.splval) ]
    udf.df = pd.concat(udf.df, ignore_index=True)
    if alfiles:
        udf.df = udf.integrate_segments(path_p, path_w, df=udf.df, rmvnoise=True)
    opath = '{dr}/{fn}.gz'.format(dr=df_dir, fn=fname)
    udf.save_dataframe(opath, df=udf.df)
    return None

def produce_png ( obj, path_index ):
    pic_dir = create_dir(obj, 'Picture')
    numdigits = len(str(obj.number_of_frames))
    fname = obj.name(obj.paths['ult'][path_index], drop_extension=True)
    if hasattr(obj, 'raw'):
        for ind,i in enumerate(obj.raw):
            opath = '{dr}/{fn}_raw_{nm:0{wd}}.png'.format(dr=pic_dir, fn=fname, nm=ind, wd=numdigits)
            obj.save_img(path=opath, img=i)
    if hasattr(obj, 'squ'):
        for ind,i in enumerate(obj.squ):
            opath = '{dr}/{fn}_squ_{nm:0{wd}}.png'.format(dr=pic_dir, fn=fname, nm=ind, wd=numdigits)
            obj.save_img(path=opath, img=i)
    if hasattr(obj, 'fan'):
        for ind,i in enumerate(obj.fan):
            opath = '{dr}/{fn}_fan_{nm:0{wd}}.png'.format(dr=pic_dir, fn=fname, nm=ind, wd=numdigits)
            obj.save_img(path=opath, img=i)
    if hasattr(obj, 'splraw'):
        for ind,i in enumerate(obj.splraw):
            opath = '{dr}/{fn}_raw_{nm:0{wd}}_splined.png'.format(dr=pic_dir, fn=fname, nm=ind, wd=numdigits)
            obj.save_img(path=opath, img=i)
    if hasattr(obj, 'splsqu'):
        for ind,i in enumerate(obj.splsqu):
            opath = '{dr}/{fn}_squ_{nm:0{wd}}_splined.png'.format(dr=pic_dir, fn=fname, nm=ind, wd=numdigits)
            obj.save_img(path=opath, img=i)
    if hasattr(obj, 'splfan'):
        for ind,i in enumerate(obj.splfan):
            opath = '{dr}/{fn}_fan_{nm:0{wd}}_splined.png'.format(dr=pic_dir, fn=fname, nm=ind, wd=numdigits)
            obj.save_img(path=opath, img=i)
    return None

def produce_video ( obj, imgtype, videopath, audiopath, outpath ):
    if videopath == outpath:
        raise ValueError('videopath and outpath must be different')
    imgs = getattr(obj, imgtype)
    obj.to_video(videopath, imgs=imgs)

    ff = obj.timeinsecsoffirstframe
    a_temppath = obj.parent(outpath) + '/audio_temp.wav'
    cmd = ['sox', audiopath, a_temppath, 'trim', str(ff), 'rate', '16000']
    subprocess.call(cmd)

    cmd = ['ffmpeg', '-i', videopath, '-i', a_temppath, '-c:v', 'copy', '-c:a', 'aac', outpath]
    subprocess.call(cmd)
    os.remove(videopath)
    os.remove(a_temppath)
    return None

def produce_avi ( obj, path_index ):
    vid_dir = create_dir(obj, 'Videos')
    fname = obj.name(obj.paths['ult'][path_index], drop_extension=True)
    apath = obj.paths['wav'][path_index]
    imgtypes = ['raw','squ','fan']
    for i in imgtypes:
        vpath = '{}/{}_{}_temp.avi'.format(vid_dir, fname, i)
        opath = '{}/{}_{}.avi'.format(vid_dir, fname, i)
        try:
            produce_video(obj, i, vpath, apath, opath)
        except:
            pass
    return None

def produce_wrapper (obj, indx, dfT, picT, vidT, crp, rsl, imgtype, flip_directions, cores, spl):
    obj = prepare_imgs(obj, indx)
    obj = crop_local(obj, crp)
    obj = reduce_resolution_local(obj, rsl)
    obj = spline_local(obj, spl, cores)
    obj = change_img_shapes(obj, imgtype, cores)
    obj = flip_local(obj, flip_directions)
    if dfT:
        produce_df(obj, indx)
    if picT:
        produce_png(obj, indx)
    if vidT:
        produce_avi(obj, indx)
    return None
### Produce functions ###

### Main functions ###
def main ( obj ) :
    obj = ask_target_dir(obj)
    whattodo = ask_whattodo()
    picT, vidT, dfT, tgT = whattodo_to_flags(whattodo)
    if tgT:
        check_paths_alright(obj, obj.wdir)
        produce_textgrids(obj)
    if any([dfT, picT, vidT]):
        check_paths_alright(obj, obj.wdir)
        if any([picT, vidT]):
            imgtype = ask_which_imgtype()
            cores = determine_cores(imgtype)
        else:
            imgtype = None
            cores = 1
        crp = where_to_crop()
        rsl = is_resol_reduc_needed()
        flip_directions = determine_flip_directions()
        spl = ask_spline_fitting(boolean=True)
        if spl and not any([picT, vidT]):
            cores = ask_cores()
        for i in tqdm(range(len(obj.paths['ult'])), desc='Main'):
            produce_wrapper(obj, i, dfT, picT, vidT, crp, rsl, imgtype, flip_directions, cores, spl)
    return None
### Main functions ###

### Body ###
if __name__ == '__main__':
    main(upc)
###


### ARCHIVES ###
# def which_img_type ():
#     print('\n##########')
#     print('- What shape of pictures do you want?')
#     print('--- Raw (rectangle) => raw')
#     print('--- Square          => squ')
#     print('--- Fan-shape       => fan')
#     print('--- All             => all')
#     opts = [ 'raw', 'squ', 'fan', 'all' ]
#     ok = False
#     while not ok:
#         inpt = input()
#         if inpt in opts:
#             ok = True
#         else:
#             print('- Please choose out of the followings:')
#             print('--- {}'.format(', '.join(opts)))
#     return inpt
# 
# def produce_pictures ( obj, picture, video, imgtype, flip_directions, wheretocrop, resolreduc, cores):
#     print('\n##########')
#     print('Producing pictures...')
#     for i in tqdm(range(len(obj.paths['ustxt']))):
#         obj = prepare_imgs(obj, i)
#         if not wheretocrop is None:
#             obj.crop(wheretocrop, inplace=True)
#         if not resolreduc is None:
#             obj.reduce_resolution(every_y=resolreduc, inplace=True)
#         obj = change_img_shapes(obj, imgtype, cores)
#         if not flip_directions is None:
#             obj = flip_wrapper(obj, flip_directions)
#         fname = obj.name(obj.paths['txt'][i], drop_extension=True)
#         if video:
#             vid_dir = create_dir(obj, 'Videos')
#             apath = obj.paths['wav'][i]
#             vpath = '{}/{}_temp.avi'.format(vid_dir, fname)
#             opath = '{}/{}.avi'.format(vid_dir, fname)
#             produce_video(obj, vpath, apath, opath)
#         if picture:
#             pic_dir = create_dir(obj, 'Picture')
#             if hasattr(obj, 'raw'):
#                 save_pics(obj, 'raw', pic_dir, fname)
#             if hasattr(obj, 'squ'):
#                 save_pics(obj, 'squ', pic_dir, fname)
#             if hasattr(obj, 'fan'):
#                 save_pics(obj, 'fan', pic_dir, fname)
#     return None
# 
# def produce_df ( obj, pr_tg=None, rmvnoise=False ):
#     if pr_tg is None:
#         print('\n##########')
#         print('- Do you want to run Aligner (yes or no)')
#         print('--- You can integrate alignment information in the dataframe.')
#         print('--- Type no, if...')
#         print('      you do not want that, or...')
#         print('      you already have those files, e.g. ***.phoneswithQ')
#         pr_tg = input()
# 
#     if not pr_tg in ['yes', 'no']:
#         raise ValueError('yes or no is acceptable.')
#     if pr_tg == 'yes':
#         produce_textgrids(obj)
# 
#     df_dir = create_dir(obj, 'Dataframes')
#     obj.set_paths(obj.wdir)
#     print('\n##########')
#     print('Producing dataframes...')
#     for i in tqdm(range(len(obj.paths['ustxt']))):
#         obj = prepare_imgs(obj, i)
#         udf = pyult.UltDf(obj)
#         udf.img_to_df(inplace=True, add_time=True)
#         try:
#             path_p = udf.paths['phoneswithQ'][i]
#             path_w = udf.paths['words'][i]
#             udf.integrate_segments(path_p, path_w, inplace=True, rmvnoise=rmvnoise)
#         except IndexError:
#             pass
#         fname = udf.name(udf.paths['txt'][i], drop_extension=True)
#         opath = '{dr}/{fn}.gz'.format(dr=df_dir, fn=fname)
#         udf.save_dataframe(opath)
#     return None
# 
# def ask_rmv_noise ():
#     print('\n##########')
#     print("- Do you want to remove marginal segment/word rows, e.g. '<P>', '_p:_', '' (empty)?")
#     print("- yes or no is acceptable")
#     isok = False
#     while not isok:
#         rmvnoise = input()
#         isok = check_yes_or_no(rmvnoise)
#     return yesno_boolean(rmvnoise)
# 
# def yesno_boolean ( yesno ):
#     isyes = yesno.lower() in ['yes','y']
#     return isyes
# 
# picT, vidT, dfT, tgT = whattodo_to_flags(whattodo)
# 
# if picT or vidT:
#     whichimgtype = which_img_type()
#     flip_directions = determine_flip_directions()
#     wheretocrop = where_to_crop()
#     resolreduc = is_resol_reduc_needed()
#     cores = determine_cores(whichimgtype)
#     produce_pictures(upc, picT, vidT, whichimgtype, flip_directions, wheretocrop, resolreduc, cores)
# 
# if tgT: produce_textgrids(upc)
# 
# if dfT:
#     rmvnoise = ask_rmv_noise()
# 
# if dfT and tgT:
#     produce_df(upc, pr_tg='no', rmvnoise=rmvnoise)
# elif dfT and not tgT:
#     produce_df(upc, rmvnoise=rmvnoise)
# 
# if not any([picT,vidT,dfT,tgT]):
#     errtxt = 'pic, vid, df, tg, or a combination of them is acceptable.\n'
#     errtxt = errtxt + 'You typed {}'.format(whattodo)
#     raise ValueError(errtxt)
