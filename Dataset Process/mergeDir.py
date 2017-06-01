#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : MIN
 PURPOSE : merge same first dir name
 VERSION : 0.2
 DATE : 4.2017
"""

__author__ = 'Min'

import os.path
import shutil
import tensorflow as tf
from tensorflow.python.platform import gfile

IMAGE_DIR = '/mnt/125_Flowers_Small_Pictures'
TO_DIR = '/mnt/Merged_125'



if __name__ == '__main__':
    if not gfile.Exists(IMAGE_DIR):
        print("Image dir'" + IMAGE_DIR +"'not found.'")

    subDirs = [x[0] for x in gfile.Walk(IMAGE_DIR)]
    count = 0
    lastName = ''
    isRootDir = True
    for subDir in subDirs:
        if isRootDir:
            isRootDir = False
            continue
        print('subDir Name is: ', subDir)
        thisName = subDir.split('/')[-1].split(' ')[0]
        print('The first Name of the flower is: ', thisName)
        toDirName = os.path.join(TO_DIR, subDir.split('/')[-1].split(' ')[0])
        if not os.path.exists(toDirName):
            os.makedirs(toDirName)

        if thisName != lastName:
            count = 0
        for imageName in os.listdir(subDir):
            imagePath = os.path.join(subDir, imageName)
            shutil.copy(imagePath, toDirName)
            newName = str(count) + '.jpg'
            count += 1
            os.rename(os.path.join(toDirName, imageName),
                      os.path.join(toDirName, newName))
        lastName = thisName
    print('ALL DONE')
