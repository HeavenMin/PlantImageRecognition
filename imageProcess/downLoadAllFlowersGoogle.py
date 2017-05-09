#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : MIN
 PURPOSE : practice download flowers form Google
 VERSION : 0.1
 DATE : 3.2017
"""

__author__ = 'Min'

import os
import sys
from icrawler.builtin import GoogleImageCrawler

image_path = sys.argv[1]

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
    google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                        storage={'root_dir' : imageDir})
    google_crawler.crawl(keyword= name, max_num=1000,
                         date_min=None, date_max=None,
                         min_size=(200,200), max_size=None)
