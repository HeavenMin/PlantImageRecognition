
"""
 AUTHOR : Lang
 PURPOSE : Multi Self Deep Learning
"""

__author__ = 'Lang'

import os
from shutil import copyfile

for dir in os.listdir('.'):
    try:
        for file in os.listdir(dir):
            if '.jpg' in file:
                print(file)
                try:
                    os.stat('/mnt/125_Copy/'+ dir)
                except:
                    os.mkdir('/mnt/125_Copy/'+ dir)
                copyfile(dir+"/"+file,'/mnt/125_Copy/'+ dir + '/' + file)
    except:
        print('error')
