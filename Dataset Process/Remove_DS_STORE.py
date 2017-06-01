
"""
 AUTHOR : Lang
 PURPOSE : Multi Self Deep Learning
"""

__author__ = 'Lang'

import os

for dir in os.listdir('.'):
    try:
        print(dir)
        for file in os.listdir(dir):
            if "DS_Store" in file:
                os.remove(dir+"/"+file)
    except:
        print('error')
