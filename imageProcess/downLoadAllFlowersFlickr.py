#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : MIN
 PURPOSE : practice download flowers form Flickr
 VERSION : 0.1
 DATE : 3.2017
"""

__author__ = 'Min'

import os
import sys
from datetime import date
from icrawler.builtin import FlickrImageCrawler

image_path = sys.argv[1]
API_KEY = '6443fc1e493dab0e64443981f7364370'

flowerWebSiteFile1 = "flowerNameSet1.html"
flowerWebSiteFile2 = "flowerNameSet2.html"

def getFlowerName(htmlFile):
    flowerNameList = []
    with open(htmlFile, "r", encoding="UTF-8") as f:
        for line in f:
            if "alt=" in line:
                info = line[(line.index("alt=") + 5):]
                flowerName = info[:(info.index("image") - 1)]
                flowerNameList.append(flowerName)
    return flowerNameList

flowerList1 = getFlowerName(flowerWebSiteFile1)
flowerList2 = getFlowerName(flowerWebSiteFile2)
flowerList = flowerList1 + flowerList2

for name in flowerList:
    imageDir = image_path + '/' + name
    flickr_crawler = FlickrImageCrawler(API_KEY, storage={'root_dir': imageDir})
    flickr_crawler.crawl(max_num = 4000, tags = name)
