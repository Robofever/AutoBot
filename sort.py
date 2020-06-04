from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from operator import itemgetter 
import shutil
import os
import sys
from time import sleep
import progressbar

path = os.path.abspath(os.getcwd())
path += "\PanelImages"

image_names = [f for f in listdir(path) if isfile(join(path, f))]

# a = image_names[0].split('_')[-3]
# print(a)
# ['solar_Fri_Jun_16_10__0__11_2017_L_0.906153208302_I_0.321592156863.jpg', 'solar_Fri_Jun_16_10__0__16_2017_L_0.903081697073_I_0.293192156863.jpg', 'solar_Fri_Jun_16_10__0__1_2017_L_0.916698044034_I_0.39577254902.jpg', 'solar_Fri_Jun_16_10__0__21_2017_L_0.903081697073_I_0.293192156863.jpg', 'solar_Fri_Jun_16_10__0__26_2017_L_0.896087391118_I_0.27462745098.jpg']
# 0.906153208302

# print(len(image_names))
# print(image_names[:5])

split_names = []

for i in range(0,len(image_names)):
    split_names.append({'name': image_names[i],'value': float(image_names[i].split('_')[-3])})

# print(split_names[:5])

new_l = sorted(split_names,key = itemgetter('value'))

# print(new_l[:5])

names_of_sorted_files = []
sorted_values = []
for dic in new_l:
    names_of_sorted_files.append(dic['name'])
    sorted_values.append(dic['value'])

# print(names_of_sorted_files[:5])
# print(sorted_values[:5])

for i in range(1,11):
    os.mkdir(os.path.join(path,f"{round(i/10,1)}"))

def copy(original,target):
    shutil.copy(original,target)
    return None

def decide(loss,name,path=path):
    target = path
    path = os.path.join(path,name)
    if loss <= 0.1:
        return copy(path,os.path.join(target,r"0.1"))
    elif loss <= 0.2:
        return copy(path,os.path.join(target,r"0.2"))
    elif loss <= 0.3:
        return copy(path,os.path.join(target,r"0.3"))
    elif loss <= 0.4:
        return copy(path,os.path.join(target,r"0.4"))
    elif loss <= 0.5:
        return copy(path,os.path.join(target,r"0.5"))
    elif loss <= 0.6:
        return copy(path,os.path.join(target,r"0.6"))
    elif loss <= 0.7:
        return copy(path,os.path.join(target,r"0.7"))
    elif loss <= 0.8:
        return copy(path,os.path.join(target,r"0.8"))
    elif loss <= 0.9:
        return copy(path,os.path.join(target,r"0.9"))
    elif loss <= 1.0:
        return copy(path,os.path.join(target,r"1.0"))

bar = progressbar.ProgressBar(maxval=len(new_l), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

bar.start()
for i in range(0,len(new_l)):
    loss = sorted_values[i]
    name = names_of_sorted_files[i]
    decide(loss,name)
    bar.update(i+1)
    sleep(0.00000000000000000000000000000000000000000000000000000000000001)
bar.finish()