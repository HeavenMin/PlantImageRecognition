#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : MIN
 PURPOSE : practice for del fake jpeg photo
 VERSION : 0.1
 DATE : 3.2017
"""

__author__ = 'Min'


import os
import imghdr

for (dirpath, dirnames, filenames) in os.walk("/Users/heaven/Desktop/GRADUATION_PROJECT/flowerImages"):
    for filename in filenames:
        if imghdr.what(dirpath + '/' +filename) != 'jpeg':
            print(filename)
            os.remove(dirpath + '/' + filename)
