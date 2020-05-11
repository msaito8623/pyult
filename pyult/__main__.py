import pyult
upc = pyult.UltPicture()
import argparse
parser = argparse.ArgumentParser()
parser.parse_args()
import os
import pdb
import subprocess
import shutil

def check_all_set ():
    if not upc.is_all_set(exclude_empty_list=True):
        upc.is_all_set(exclude_empty_list=True, verbose=True)
        raise ValueError('You do not have all the necessary files.')
    return None

def what_to_do ():
    print('\n###')
    print('What do you want from the file(s)?')
    print('Pictures => pic')
    print('Videos => vid')
    print('Dataframes => df')
    print('TextGrids => tg')
    print('You can specify them together with a space, e.g. pic vid')
    print('If you want them all => all')
    return input()

def which_img_type ():
    print('\n###')
    print('What shape of pictures do you want?')
    print('Raw (rectangle) => raw')
    print('Square => squ')
    print('Fan-shape => fan')
    inpt = input()
    opts = [ 'raw', 'squ', 'fan' ]
    if not inpt in opts:
        raise ValueError('raw, squ, or fan is acceptable.')
    return inpt

def create_dir ( obj, dirname ):
    newdir = obj.wdir + '/' + dirname
    os.makedirs(newdir, exist_ok=True)
    return newdir

def prepare_imgs ( obj, ind ):
    obj.read(obj.paths['ult'][ind], inplace=True)
    obj.read(obj.paths['ustxt'][ind], inplace=True)
    obj.vec_to_pics(inplace=True)
    return obj

def change_img_shapes ( obj, imgtype ):
    if imgtype=='squ':
        obj.to_square(inplace=True)
    elif imgtype=='fan':
        obj.fanshape(inplace=True, numvectors=obj.img.shape[-1], magnify=4)
    return obj

def determine_flip_directions ():
    opts = [ 'yes', 'no' ]
    print('\n###')
    print('Do you want to flip pictures horizontally? (yes or no)')
    xflip = input()
    if not xflip in opts:
        raise ValueError('yes or no is acceptable.')
    print('\n###')
    print('Do you want to flip pictures vertically? (yes or no)')
    yflip = input()
    if not yflip in opts:
        raise ValueError('yes or no is acceptable.')
    if xflip=='yes' and yflip=='yes':
        flp = 'xy'
    elif xflip=='yes' and yflip=='no':
        flp = 'x'
    elif xflip=='no' and yflip=='yes':
        flp = 'y'
    else:
        flp = None
    return flp

def where_to_crop ():
    opts = [ 'yes', 'no' ]
    print('\n###')
    print('Do you want to crop pictures? (yes or no)')
    cropornot = input()
    if not cropornot in opts:
        raise ValueError('yes or no is acceptable.')
    crp = None
    if cropornot == 'yes':
        print('\n###')
        print('Where to crop? Specify as a 4-element tuple, i.e. (xmin, xmax, ymin, ymax).')
        print('Please type None if you want to refer to the ends of x/y-axes.')
        print('E.g. "You want to cut off only the first 10 pixels along x-axis" ==> (10, None, None, None)')
        crp = input()
        tgt = [ ' ', '(', ')' ]
        for i in tgt:
            crp = crp.replace(i, '')
        crp = tuple(map(lambda x: int(x) if x!='None' else None, crp.split(',')))
    return crp

def is_resol_reduc_needed ():
    opts = [ 'yes', 'no' ]
    print('\n###')
    print('Do you want to reduce resolution of y-axis? (yes or no)')
    print('Because ultasound pictures have much more information along y-axis than x-axis, it might not hurt to take every n-th pixel along y-axis to reduce data size.')
    resolreduc = input()
    if not resolreduc in opts:
        raise ValueError('yes or no is acceptable.')
    nth = None
    if resolreduc == 'yes':
        print('\n###')
        print('How much to reduce?')
        print('E.g. 3, meaning you take every 3rd pixel along y-axis, leading to approximately 1/3 of the original data size.')
        nth = input()
        try:
            nth = int(nth)
        except ValueError:
            raise ValueError('Only integers are acceptable.')
    return nth

def produce_pictures ( obj, picture, video, imgtype, flip_directions, wheretocrop, resolreduc):
    for i in range(len(obj.paths['ustxt'])):
        obj = prepare_imgs(obj, i)
        if not wheretocrop is None:
            obj.crop(wheretocrop, inplace=True)
        if not resolreduc is None:
            obj.reduce_resolution(every_y=resolreduc, inplace=True)
        obj = change_img_shapes(obj, imgtype)
        if not flip_directions is None:
            obj.flip(flip_directions, inplace=True)
        fname = obj.name(obj.paths['txt'][i], drop_extension=True)
        if video:
            vid_dir = create_dir(upc, 'Videos')
            apath = obj.paths['wav'][i]
            vpath = '{}/{}_temp.avi'.format(vid_dir, fname)
            opath = '{}/{}.avi'.format(vid_dir, fname)
            produce_video(obj, vpath, apath, opath)
        if picture:
            pic_dir = create_dir(upc, 'Picture')
            for j in range(len(obj.img)):
                numdigits = len(str(obj.number_of_frames))
                opath = '{dr}/{fn}_{im}_{nm:0{wd}}.png'.format(dr=pic_dir, fn=fname, im=imgtype, nm=j, wd=numdigits)
                obj.save_img(path=opath, img=obj.img[j])
    return None

def produce_video ( obj, videopath, audiopath, outpath ):
    if videopath == outpath:
        raise ValueError('videopath and outpath must be different')
    obj.to_video(videopath)

    ff = obj.timeinsecsoffirstframe
    a_temppath = obj.parent(outpath) + '/audio_temp.wav'
    cmd = ['sox', audiopath, a_temppath, 'trim', str(ff), 'rate', '16000']
    subprocess.call(cmd)

    cmd = ['ffmpeg', '-i', videopath, '-i', a_temppath, '-c:v', 'copy', '-c:a', 'aac', outpath]
    subprocess.call(cmd)
    os.remove(videopath)
    os.remove(a_temppath)
    return None

def produce_df ( obj, pr_tg=None):
    if pr_tg is None:
        print('\n###')
        print('Do you want to run Aligner to integrate alignment information in the dataframe? (yes or no)')
        print('If you do not want that, or if you already have those files (e.g. ***.phoneswithQ), type no.')
        pr_tg = input()

    print('\n###')
    print('How many cores do you want to use? Producing dataframes may take a while.')
    cores = input()

    if not pr_tg in ['yes', 'no']:
        raise ValueError('yes or no is acceptable.')
    if pr_tg == 'yes':
        produce_textgrids(obj)

    try:
        cores = int(cores)
    except ValueError:
        print('\n###')
        print('WARNING: The input cannot be interpreted as an integer value. cores=1 is used instead.')
        cores = 1
    df_dir = create_dir(obj, 'Dataframes')
    obj.set_paths(obj.wdir)
    for i in range(len(obj.paths['ustxt'])):
        obj = prepare_imgs(obj, i)
        udf = pyult.UltDf(obj)
        udf.img_to_df(inplace=True, add_time=True)
        try:
            path_p = udf.paths['phoneswithQ'][i]
            path_w = udf.paths['words'][i]
            udf.integrate_segments(path_p, path_w, inplace=True, cores=cores)
        except IndexError:
            pass
        fname = udf.name(udf.paths['txt'][i], drop_extension=True)
        opath = '{dr}/{fn}.gz'.format(dr=df_dir, fn=fname)
        udf.save_dataframe(opath)
    return None

def produce_textgrids ( obj ):
    print('\n###')
    print('Where is the home directory for Aligner? (E.g. /home/username/Aligner)')
    print('As default, it should contain AUTHORS, bin, BUGS, confs, dict, ...and so on.')
    alhome = input()

    print('\n###')
    print('Do you want to keep temporary files (yes or no)')
    print('E.g. A wav file whose sampling rate is 16kHz. The produced TextGrid file is based on this audio file. You should open it with its corresponding TextGrid file if you want to align and display them in Praat.')
    keeptmp = input()

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


### Body ###
print('\n###')
print('Give me a file path or a directory path which has all the necessary files.')
inpt = input()
upc.set_paths(inpt)
check_all_set()

whattodo = what_to_do()
picT = 'pic' in whattodo
vidT = 'vid' in whattodo
dfT = 'df' in whattodo
tgT = 'tg' in whattodo
if whattodo=='all': picT,vidT,dfT,tgT = [True]*4

if picT or vidT:
    whichimgtype = which_img_type()
    flip_directions = determine_flip_directions()
    wheretocrop = where_to_crop()
    resolreduc = is_resol_reduc_needed()
    produce_pictures(upc, picT, vidT, whichimgtype, flip_directions, wheretocrop, resolreduc)

if tgT: produce_textgrids(upc)

if dfT and tgT:
    produce_df(upc, pr_tg='no')
elif dfT and not tgT:
    produce_df(upc)

if not any([picT,vidT,dfT,tgT]):
    errtxt = 'pic, vid, df, tg, or a combination of them is acceptable.\n'
    errtxt = errtxt + 'You typed {}'.format(whattodo)
    raise ValueError(errtxt)



