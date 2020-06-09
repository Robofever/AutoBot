import os
import sys
from os import listdir
from os.path import isfile, join , abspath
import shutil
from tqdm import tqdm

path = abspath(os.getcwd())
fpath = join(path,r"PanelImages")

images = [f for f in listdir(fpath) if isfile(join(fpath, f))]

clean_images = []
with open(join(path,r"Clean.txt")) as clean:
    clean_images = (clean.read()).split(sep="\n")

# print(clean_images)
os.mkdir(join(path,r"Clean"))
os.mkdir(join(path,r"Unclean"))

pbar = tqdm(total=len(images))

for name in images:
    if name in clean_images:
        shutil.move(join(fpath,name),join(path,r"Clean"))
    else :
        shutil.move(join(fpath,name),join(path,"Unclean"))
    pbar.update()
pbar.close()
print("Done")
