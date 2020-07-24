import argparse
parser = argparse.ArgumentParser(description='Produces an image with a fitted spline curve from an ultrasound image.')
parser.add_argument('path', help='Path to the image file you want to add a spline curve.')
parser.add_argument('--cores', type=int, help='How many cores to use.')
parser.add_argument('--value', action='store_true', help='Returns fitted y-axis values of a fitted spline curve if specified.')
args = parser.parse_args()

import pandas as pd
import pyult.pyult as pyult
import pyult.splfit as splfit
from pathlib import Path

upc = pyult.UltPicture()
img = upc.read_img(args.path)


if args.value:
    val = splfit.fit_spline(img, cores=args.cores)
    val = pd.DataFrame(val)
else:
    img = splfit.fit_spline_img(img, cores=args.cores)


pdir = str(Path(args.path).parent)
fname = str(Path(args.path).stem)
if args.value:
    sfx = '.csv'
else:
    sfx = str(Path(args.path).suffix)
opath = '{}/{}_spline{}'.format(pdir, fname, sfx)

flg = True
cnt = 1
while flg:
    if Path(opath).exists():
        opath = '{}/{}_spline-{}{}'.format(pdir, fname, cnt, sfx)
        cnt = cnt + 1
    else:
        flg = False

if args.value:
    val.to_csv(opath, sep='\t', index=False, header=True)
else:
    upc.save_img(path=opath, img=img)
