#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : MIN
 PURPOSE : practice for del not jpeg photo
 VERSION : 0.1
 DATE : 3.2017
"""

__author__ = 'Min'

import os
import shutil


for (dirpath, dirnames, filenames) in os.walk("/Users/heaven/Desktop/GRADUATION_PROJECT/flowerImages"):
    for filename in filenames:
        if not filename.endswith('.jpg'):
            print(filename)
            os.remove(dirpath + '/' + filename)
    for dirname in dirnames:
        if dirname.startswith('0'):
            shutil.rmtree(dirpath + '/' + dirname)
