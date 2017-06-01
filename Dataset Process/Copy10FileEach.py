
"""
 AUTHOR : Lang
 PURPOSE : Multi Self Deep Learning
"""

__author__ = 'Lang'

import os
from shutil import copyfile

for dir in os.listdir('.'):
    try:
        count = 0
        for file in os.listdir(dir):
            if '.jpg' in file:
                count = count + 1
                copyfile(dir+"/"+file,'/Users/langlin/work/Year2/Project/flower_or_not/Yes/'+ file)
            if count >= 10:
                break
    except:
        print('error')
