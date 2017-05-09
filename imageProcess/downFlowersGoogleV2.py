#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : MIN
 PURPOSE : download flower image and fliter fake jpeg
 VERSION : 0.1
 DATE : 4.2017
"""

__author__ = 'Min'

import os
import sys
import shutil
import imghdr
from icrawler.builtin import GoogleImageCrawler

flowerNameSetFile_path = sys.argv[1]
image_path = sys.argv[2]

def getFlowerName(path):
    flowerNameList = []
    with open(path, "r", encoding="UTF-8") as f:
        for flowerName in f:
            flowerNameList.append(flowerName[:-1])
    return flowerNameList

def downloadFlowers(flowerList ,image_path):
    for name in flowerList:
        imageDir = image_path + '/' + name
        google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                            storage={'root_dir' : imageDir})
        google_crawler.crawl(keyword= name, max_num=1000,
                             date_min=None, date_max=None,
                             min_size=(200,200), max_size=None)

def delNotJpeg(path):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if not filename.endswith('.jpg'):
                os.remove(dirpath + '/' + filename)
        for dirname in dirnames:
            if dirname.startswith('0'):
                shutil.rmtree(dirpath + '/' + dirname)

def delFakeJpeg(path):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if imghdr.what(dirpath + '/' +filename) != 'jpeg':
                os.remove(dirpath + '/' + filename)

if __name__ == '__main__':
    flowerList = getFlowerName(flowerNameSetFile_path)
    downloadFlowers(flowerList, image_path)
    delNotJpeg(image_path)
    delFakeJpeg(image_path)
