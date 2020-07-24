import argparse
parser = argparse.ArgumentParser(description='Produces an image with a fitted spline curve from an ultrasound image.')
parser.add_argument('path', help='Path to the image file you want to add a spline curve.')
args = parser.parse_args()

import pyult.pyult as pyult
import pyult.splfit as splfit
from pathlib import Path

upc = pyult.UltPicture()
img = upc.read_img(args.path)
img = splfit.fit_spline_img(img)
pdir = str(Path(args.path).parent)
fname = str(Path(args.path).stem)
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
upc.save_img(path=opath, img=img)
