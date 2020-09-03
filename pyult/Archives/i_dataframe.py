import argparse
parser = argparse.ArgumentParser(description='Produces a dataframe from an (ultrasound) image.')
parser.add_argument('path', help='Path to the image file you want to make a dataframe from.')
parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrites the file with the same name.')

args = parser.parse_args()

import pandas as pd
import pyult.pyult as pyult
import pyult.splfit as splfit
from pathlib import Path

upc = pyult.UltPicture()
img = upc.read_img(args.path)
img = upc.flip(flip_direction='y', img=img)

udf = pyult.UltDf()
dat = udf.img_to_df(img=img)

pdir = str(Path(args.path).parent)
fname = str(Path(args.path).stem)
sfx = '.gz'
opath = '{}/{}_dataframe{}'.format(pdir, fname, sfx)

flg = not args.overwrite
cnt = 1
while flg:
    if Path(opath).exists():
        opath = '{}/{}_dataframe-{}{}'.format(pdir, fname, cnt, sfx)
        cnt = cnt + 1
    else:
        flg = False

dat.to_csv(opath, sep='\t', index=False, header=True, compression='gzip')
