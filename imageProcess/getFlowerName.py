#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : MIN
 PURPOSE : get flowerName from a html file
 VERSION : 0.1
 DATE : 3.2017
"""

__author__ = 'Min'

import os

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

def writeAText(flowerNameList):
    with open('125FlowerNames.txt', 'w') as f:
        for flowerName in flowerNameList:
            f.write(flowerName + '\n')

# for name in flowerList:
#     os.mkdir(name)

if __name__ == '__main__':
    flowerList1 = getFlowerName(flowerWebSiteFile1)
    flowerList2 = getFlowerName(flowerWebSiteFile2)
    flowerList = flowerList1 + flowerList2
    writeAText(flowerList)
