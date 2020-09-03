import argparse
parser = argparse.ArgumentParser(description='Produces a fan-shaped image from a raw ultrasound image.')
parser.add_argument('path', help='Path to the raw ultrasound image file you want to convert to a fanshaped.')
parser.add_argument('-par', '--parameter', help='Path to the parameter file corresponding to the image provided by "path" (the first positional argument), e.g. xxxxxUS.txt')
parser.add_argument('-mag', '--magnify', type=float, help='How much to magnify the image. This value is recommended to be set larger when using a corresponding parameter file.')
args = parser.parse_args()

import pyult.pyult as pyult
import pyult.splfit as splfit
from pathlib import Path

upc = pyult.UltPicture()
upc.read_img(args.path, inplace=True, grayscale=False)

if not args.parameter is None:
    upc.read(args.parameter, inplace=True)

magnify = 1 if args.magnify is None else args.magnify
upc.fanshape_2d(inplace=True, magnify=magnify)

pdir = str(Path(args.path).parent)
fname = str(Path(args.path).stem)
sfx = str(Path(args.path).suffix)
opath = '{}/{}_fan{}'.format(pdir, fname, sfx)

flg = True
cnt = 1
while flg:
    if Path(opath).exists():
        opath = '{}/{}_fan-{}{}'.format(pdir, fname, cnt, sfx)
        cnt = cnt + 1
    else:
        flg = False
upc.save_img(path=opath)
