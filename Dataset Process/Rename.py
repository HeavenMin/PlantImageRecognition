
"""
 AUTHOR : Lang
 PURPOSE : Multi Self Deep Learning
"""

__author__ = 'Lang'

import os

for dir in os.listdir('.'):
    try:
        print(dir)
        count = 0
        for file in os.listdir(dir):
            if ".jpg" in file:
                count = count + 1
                os.rename(dir+"/"+file,dir+"/"+dir+" "+str(count)+".jpg")
    except:
        print('error')
