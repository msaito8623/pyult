import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='Path to a directory, which should have all the relevant files, e.g. xxxxUS.txt')
parser.add_argument('-t', '--task', help='What to do. Specify either pic (png images), vid (videos), df (dataframes), tg (TextGrids), all, or combinations of them with commas but no spaces, e.g. pic,tg,df.')
parser.add_argument('-i', '--interactive', action='store_false', help='If provided, the interactive mode is turned off. With the interactive option on, the script asks you what to do for the options unspecified.')
parser.add_argument('-cr', '--crop', help='Cropping points on all the four sides of images. Specify them as XMIN,XMAX,YMIN,YMAX, e.g. 20,50,100,600')
parser.add_argument('-r', '--resolution', help='How much to reduce resolution along y-axis. For example, 3 takes every 3rd pixel along y-axis in each frame.')
parser.add_argument('-x', '--flipx', action='store_true', help='If provided, each image is flipped horizontally.')
parser.add_argument('-y', '--flipy', action='store_true', help='If provided, each image is flipped vertically.')
parser.add_argument('-cb', '--combine', action='store_true', help='If provided, output dataframes will be combined into a single gz file. Otherwise, dataframes are produced for each recording.')
parser.add_argument('-s', '--spline', action='store_true', help='If provided, spline curves are fitted to each frame and included in the products, e.g. dataframe.')
parser.add_argument('-co', '--cores', help='How many cores to be used. Parallel processing is carried out with multiple cores for fitting splines and making fan-shaped pictures.')
parser.add_argument('-ft', '--figuretype', help='What shapes of images should be produced: raw (rectangle), squ (square), fan (fan-shaped). Combinations of the options (connected by commas) are also acceptable, e.g. fan,raw')
parser.add_argument('-cl', '--clean', action='store_true', help='If provided, the target directory specified by --path is cleaned, so all the problematic files are removed and put into a new directory named "BadFiles" in the same directory as the target directoryl.')
args = parser.parse_args()
import copy
import os
import pandas as pd
import pdb
import pyult.pyult as pyult
import pyult.splfit as splfit
import shutil
import subprocess
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
def ask_target_dir (obj, path, interactive, clean):
    if path is None:
        if interactive:
            print('\n##########')
            print('- Give me a file path or a directory path which has all the necessary files.')
            tdir = input()
        else:
            raise ValueError('Path to a directory is not provided.')
    else:
        tdir = path
    if clean:
        obj.clean_dir(tdir)
    ok = False
    while not ok:
        if obj.exist(tdir):
            obj.set_paths(tdir)
            if obj.is_all_set(exclude_empty_list=True):
                ok = True
            else:
                print('- The directory specified does not have all necessary files ready. Please try again.')
                tdir = input()
        else:
            print('- The path specified does not exist. Please try again.')
            tdir = input()
    return obj

def ask_whattodo (task, interactive):
    def print_texts ():
        print('\n##########')
        print('- What do you want from the file(s)?')
        print('--- Pictures             => pic')
        print('--- Videos               => vid')
        print('--- Dataframes           => df')
        print('--- TextGrids            => tg')
        print('--- If you want them all => all')
        print('--- You can specify them together with a comma but no space, e.g. pic,vid')
        return None
    def ok_ng_check (task, ok_inputs):
        task = task.split(',')
        task = [ i for i in task if i != '' ]
        oks = [ i for i in task if     i in ok_inputs ]
        ngs = [ i for i in task if not i in ok_inputs ]
        return (oks, ngs)
    ok_inputs = [ 'pic', 'vid', 'df', 'tg', 'all' ]
    if task is None:
        if interactive:
            print_texts()
            task = input()
        else:
            raise ValueError('Specify what the script should do, by "--task" (or "-t")')
    ok_while = False
    while not ok_while:
        oks, ngs = ok_ng_check(task, ok_inputs)
        if len(oks)==0:
            print('- Please choose out of the followings:')
            print('--- {}'.format(', '.join(ok_inputs)))
            task = input()
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
        ufl = pyult.Files()
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
def where_to_crop (crop_points, interactive):
    def ask_crop_ornot ():
        print('\n##########')
        print('- Do you want to crop pictures? (yes or no)')
        ok = False
        while not ok:
            cropornot = input()
            if check_yes_or_no(cropornot):
                ok = True
        return cropornot
    def print_texts ():
        print('\n##########')
        print('- Where to crop?')
        print('--- Specify as a 4-element tuple, i.e. (xmin, xmax, ymin, ymax).')
        print('--- Please type None if you want to refer to the ends of x/y-axes.')
        print('--- E.g. "You want to cut off only the first 10 pixels along x-axis"')
        print('      ==> (10, None, None, None)')
        return None

    if crop_points is None:
        if interactive:
            wanna_crop = yesno_to_bool(ask_crop_ornot())
            if wanna_crop:
                print_texts()
                crop_points = input()
            else:
                crop_points = False
        else:
            crop_points = False
    if crop_points:
        ok = False
        while not ok:
            tgt = [ ' ', '(', ')' ]
            for i in tgt:
                crop_points = crop_points.replace(i, '')
            crop_points = crop_points.split(',')
            try:
                crop_points = tuple(map(lambda x: int(x) if x!='None' else None, crop_points))
                ok = True
            except ValueError:
                print('- Cropping points must follow the following format: e.g. (10, None, 20, None)')
                print('- Please try again...')
                crop_points = input()
    return crop_points

def crop_local ( obj, crp ):
    if crp:
        obj.crop(crp, inplace=True)
    return obj
### Cropping ###

### Reduce Resolution ###
def is_resol_reduc_needed (resol, interactive):
    def ask_reduce_resolution_ornot ():
        print('\n##########')
        print('- Do you want to reduce resolution of y-axis? (yes or no)')
        # print('(Because ultasound pictures have much more information along y-axis than x-axis, it might not hurt to take every n-th pixel along y-axis to reduce data size.)')
        ok = False
        while not ok:
            resolreduc = input()
            if check_yes_or_no(resolreduc):
                ok = True
        return resolreduc

    def print_texts ():
        print('\n##########')
        print('- How much to reduce?')
        print('--- E.g. 3')
        print('------> meaning you take every 3rd pixel along y-axis')
        print('------> leading to approximately 1/3 of the original data size.')
        return None
    
    if resol is None:
        if interactive:
            wanna_reduce_resol = yesno_to_bool(ask_reduce_resolution_ornot())
            if wanna_reduce_resol:
                print_texts()
                resol = input()
            else:
                resol = False
        else:
            resol = False
    if resol:
        ok = False
        while not ok:
            try:
                resol = int(resol)
                ok = True
            except ValueError:
                print('- Only integers are acceptable for the degree of resolution-reduction. Please try again...')
                resol = input()
    return resol

def reduce_resolution_local ( obj, rsl ):
    if rsl:
        obj.reduce_resolution(every_y=rsl, inplace=True)
    return obj
### Reduce Resolution ###

### Change image shapes ###
def ask_which_imgtype (figtype, interactive):
    def print_texts ():
        print('\n##########')
        print('- What shape of pictures do you want?')
        print('--- Raw (rectangle) => raw')
        print('--- Square          => squ')
        print('--- Fan-shape       => fan')
        print('--- All             => all')
        print('--- You can specify them together with a space, e.g. raw squ')
        return None
    if figtype is None:
        if interactive:
            print_texts()
            figtype = input()
        else:
            figtype = 'raw'
    ok_inputs = [ 'raw', 'squ', 'fan', 'all' ]
    ok_while = False
    while not ok_while:
        figtype = figtype.split(',')
        figtype = [ i for i in figtype if i != '' ]
        oks = [ i for i in figtype if     i in ok_inputs ]
        ngs = [ i for i in figtype if not i in ok_inputs ]
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
def determine_flip_directions (flipx, flipy, interactive):
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
        xyflip = yesno_to_bool(xyflip)
        return xyflip
    def to_flags ( xflip, yflip ):
        if xflip and yflip:
            flp = 'xy'
        elif xflip and not yflip:
            flp = 'x'
        elif not xflip and yflip:
            flp = 'y'
        else:
            flp = False
        return flp

    if not flipx:
        if interactive:
            flipx = ask_flip(vertical=False)
    if not flipy:
        if interactive:
            flipy = ask_flip(vertical=True)
    flp = to_flags(flipx, flipy)
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
def ask_spline_fitting(spl, interactive) :
    if not spl:
        if interactive:
            print('\n##########')
            print('- Do you want to fit spline curves? (yes or no)')
            spl = input()
            ok = False
            opts = [ 'yes', 'no', 'y', 'n']
            while not ok:
                if check_yes_or_no(spl):
                    spl = yesno_to_bool(spl)
                    ok = True
                else:
                    print('- Please choose out of the followings:')
                    print('--- {}'.format(', '.join(opts)))
                    spl = input()
    return spl

def spline_local (obj, spl, cores) :
    if spl:
        obj.splval = [ splfit.fit_spline(img=i, cores=cores) for i in tqdm(obj.img,desc='Spline') ]
        obj.splimg = [ splfit.fit_spline_img(img=i,ftv=j,cores=cores) for i,j in zip(obj.img, obj.splval) ]
    return obj
### Spline Fitting ###

### Produce functions ###
def produce_df ( obj, path_index, combine, spl ):
    df_dir = create_dir(obj, 'Dataframes')
    udf = pyult.UltDf(obj)
    try:
        path_p = udf.paths['phoneswithQ'][path_index]
        path_w = udf.paths['words'][path_index]
        alfiles = True
    except IndexError:
        try:
            path_t = udf.paths['textgrid'][path_index]
            aln = pyult.Alignment()
            opaths = aln.textgrid_to_alignfiles(path_t, return_outpaths=True)
            path_p = opaths['phoneswithQ']
            path_w = opaths['words']
            alfiles = True
        except IndexError:
            alfiles = False
            print('- WARNING: Alignment files (i.e. *.phoneswithQ and *.words) are not found, therefore segments/words information is not integrated into produced dataframes)')
    udf.df = udf.img_to_df(img=udf.raw, add_time=True, combine=False)
    if spl:
        if hasattr(udf, 'splval'):
            udf.df = [ udf.integrate_spline_values(i,j) for i,j in zip(udf.df, udf.splval) ]
        else:
            raise AttributeError('Spline values are not found in attributes.')
    udf.df = pd.concat(udf.df, ignore_index=True)
    if alfiles:
        udf.df = udf.integrate_segments(path_p, path_w, df=udf.df, rmvnoise=True)
    udf.df = udf.rmv_noise(df=udf.df)
    fname = udf.name(udf.paths['ult'][path_index], drop_extension=True)
    udf.df['filename'] = fname
    if combine:
        opath = '{dr}/combined.gz'.format(dr=df_dir)
        if path_index==0:
            dummy = pd.DataFrame(columns=udf.df.columns)
            udf.save_dataframe(opath, df=dummy, mode='w')
        udf.save_dataframe(opath, df=udf.df, mode='a', header=False)
    else:
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

def produce_wrapper (obj, indx, dfT, picT, vidT, crp, rsl, imgtype, flip_directions, cores, spl, combine):
    obj = prepare_imgs(obj, indx)
    obj = crop_local(obj, crp)
    obj = reduce_resolution_local(obj, rsl)
    obj = spline_local(obj, spl, cores)
    obj = change_img_shapes(obj, imgtype, cores)
    obj = flip_local(obj, flip_directions)
    if dfT:
        produce_df(obj, indx, combine, spl)
    if picT:
        produce_png(obj, indx)
    if vidT:
        produce_avi(obj, indx)
    return None
### Produce functions ###

### Main functions ###
def main ( obj, path, task, interactive, crop, resol, flipx, flipy, combine, spl, cores, figtype, clean ) :
    obj = ask_target_dir(obj, path, interactive, clean)
    whattodo = ask_whattodo(task, interactive)
    picT, vidT, dfT, tgT = whattodo_to_flags(whattodo)
    if tgT:
        check_paths_alright(obj, obj.wdir)
        produce_textgrids(obj)
    if any([dfT, picT, vidT]):
        check_paths_alright(obj, obj.wdir)
        if any([picT, vidT]):
            imgtype = ask_which_imgtype(figtype, interactive)
            cores = determine_cores(imgtype)
        else:
            imgtype = None
            cores = 1
        crp = where_to_crop(crop, interactive)
        rsl = is_resol_reduc_needed(resol, interactive)
        flip_directions = determine_flip_directions(flipx, flipy, interactive)
        spl = ask_spline_fitting(spl, interactive)
        maybe_mp = spl and not any([picT, vidT])
        if maybe_mp:
            if (cores is None) and interactive:
                cores = ask_cores()
            else:
                cores = 1
        for i in tqdm(range(len(obj.paths['ult'])), desc='Main'):
            produce_wrapper(obj, i, dfT, picT, vidT, crp, rsl, imgtype, flip_directions, cores, spl, combine)
    return None
### Main functions ###

### Body ###
if __name__ == '__main__':
    obj = pyult.UltPicture()
    main(obj, args.path, args.task, args.interactive, args.crop, args.resolution, args.flipx, args.flipy, args.combine, args.spline, args.cores, args.figuretype, args.clean)
###

