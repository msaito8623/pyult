import pandas as pd
from multiprocessing import Pool
from pathlib import Path
from pyult import file
from pyult import recording
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--directory', help='Path to a directory, which should have all the relevant files, e.g. xxxxUS.txt')

parser.add_argument('-v', '--verbose', action='store_true', help='If provided, problems in the specified directory are displayed (if any).')

parser.add_argument('-co', '--cores', help='How many cores to use for processing in parallel.')

parser.add_argument('-cr', '--crop', help='Cropping points on all the four sides of images. Specify them as XMIN,XMAX,YMIN,YMAX, e.g. 20,50,100,600')

parser.add_argument('-f', '--flip', help='"x", "y", or "xy" for flipping each image along x-axis, y-axis, and both for each.')

parser.add_argument('-r', '--resolution', help='How much to reduce resolution along y-axis. For example, 3 takes every 3rd pixel along y-axis in each frame.')

parser.add_argument('-s', '--spline', action='store_true', help='If provided, spline curves are fitted to each frame and included in the products, e.g. dataframe.')

args = parser.parse_args()


def main (directory, verbose, cores, crop, flip, resolution, spline):
    if not file.check_wdir(directory, verbose):
        e1 = 'The directory specified is not ready for preprocessing of ultrasound images.\n'
        e2 = 'Use --verbose, to see what is wrong in the directory.'
        raise ValueError(e1+e2)
    stems = file.unique_target_stems(directory)
    stems = [ (directory, i, crop, flip, resolution, spline) for i in stems ]
    pool = Pool(cores)
    pool.map(task, stems)
    return None

def task (par_args):
    wdir, stem, crop, flip, resol, spl = par_args
    obj = recording.Recording()
    obj.read_ult(file.find_target_file(wdir, stem, '\\.ult$'))
    obj.read_ustxt(file.find_target_file(wdir, stem, 'US\\.txt$'))
    obj.read_txt(file.find_target_file(wdir, stem, '[^S]\\.txt$'))
    obj.read_phones(file.find_target_file(wdir, stem, '\\.phones$'))
    obj.read_phones(file.find_target_file(wdir, stem, '\\.phoneswithQ$'))
    obj.read_words(file.find_target_file(wdir, stem, '\\.words$'))
    obj.read_textgrid(file.find_target_file(wdir, stem, '\\.TextGrid$'))
    obj.vec_to_imgs()
    obj.crop(crop)
    obj.flip(flip)
    obj.reduce_y(resol)
    if spl:
        obj.fit_spline(set_fitted_values=True)
    obj.imgs_to_df()
    obj.integrate_segments()
    if spl:
        obj.integrate_splines()
    opath = '{}/{}.gz'.format(wdir, stem)
    obj.df.to_csv(opath, sep='\t', index=False)
    return None

if __name__ == '__main__':
    if args.directory is None:
        raise ValueError('Please specify the target directory, where all the necessary files should be ready, e.g. xxx.ult')
    cores = 1 if args.cores is None else int(args.cores)
    main(args.directory, args.verbose, cores, args.crop, args.flip, args.resolution, args.spline)
