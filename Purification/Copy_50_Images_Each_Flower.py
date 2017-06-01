#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : Lang
 PURPOSE : download flower image from Flickr and Googel then fliter fake jpeg
 Environment : This code can only run in python3 and python2
 Usage : python "this_file_name" "image_sets_original_path" "aim_path"
"""

__author__ = 'Lang'

import os
import sys
from shutil import copyfile

original_path = sys.argv[1]
copy_path = sys.argv[2]

os_list = sorted(os.listdir(original_path))
for image_dir in os_list:
    try:
        count = 0
        images_paths = sorted(os.listdir(original_path + '/' + image_dir))
        while count < 50:
            count = count +1
            image_name = image_dir + " " + str(count) + ".jpg"
            copyfile(original_path + "/" + image_dir+"/"+image_name,copy_path + "/" + image_name)
    except:
        print('error')
