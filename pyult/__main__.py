import pyult
ult = pyult.UltPicture()
import argparse
parser = argparse.ArgumentParser()
parser.parse_args()
import sys

### Target directory ###
print('What is the target directory, where all the exported files by AAA are stored?')
tdir = input()
tdir = tdir.strip('"')
tdir = tdir.strip("'")
while not ult.exist(tdir):
    print('The path does not exist. Try again...:')
    tdir = input()
    tdir = tdir.strip('"')
    tdir = tdir.strip("'")
###

### Flip? ###
print('Do you need to flip x- or y-axis?')
ans = [ 'x', 'y', 'xy', 'No', 'no', 'NO']
flp_axs = 'default'
while not flp_axs in ans:
    print('Type x, y, or xy, if flipping is needed. Otherwise type No...:')
    flp_axs = input()
###

### Yaxis resolution? ###
print('Do you need to reduce resolution along y-axis?')
print('If yes, please type how much to reduce, e.g. 2 for every 2nd pixel.')
print('If no, please type "No".')
while not isinstance(yrsl, int):
    print('The path does not exist. Try again...:')
    tdir = input()
    tdir = tdir.strip('"')
    tdir = tdir.strip("'")
###

# ### Images necessary? ###
# print('Do you need images created?')
# img = True or False
# ###
# 
# ### Raw images necessary? ###
# print('Do you need raw images?')
# rimg = True or False
# ###
# 
# ### Square images necessary? ###
# print('Do you need squared images?')
# simg = True or False
# ###
# 
# ### Fan-shape images necessary? ###
# print('Do you need fan-shaped images?')
# fimg = True or False
# ###


