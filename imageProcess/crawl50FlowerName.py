#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : MIN
 PURPOSE : crawl http://www.stylecraze.com/articles/most-beautiful-flowers-in-the-world/#gref flower information
 VERSION : 0.1
 DATE : 5.2017
"""

__author__ = 'Min'

import requests
import re

webInfo = requests.get('http://www.stylecraze.com/articles/most-beautiful-flowers-in-the-world/#gref')
webHtml = str(webInfo.content)

flowerNameRawSet = re.findall(r'<h2>\S+ \S+:', webHtml)
flowerNameSet = []

for ele in flowerNameRawSet:
    flowerNameSet.append(ele.split(' ')[-1][:-1])

with open('50CommonFlowerName.txt', 'w') as f:
    for flowerName in flowerNameSet:
        f.write(flowerName + '\n')
