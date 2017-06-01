
"""
 AUTHOR : Lang
 PURPOSE : Multi Self Deep Learning
"""

__author__ = 'Lang'

import os;

tracing = open("After_First_Purification.txt",'w')
tracing.close()

total_count = 0
for dirs in sorted(os.listdir(".")):
    try:
        count = 0
        for files in sorted(os.listdir(dirs)):
            if ".jpg" in files:
                count = count + 1
        total_count = total_count + 1
        tracing = open("After_First_Purification.txt",'a')
        tracing.write("No." + str(total_count) + " " + dirs + " Set contains " + str(count) + " files\n")
        tracing.close()
    except:
        print("error")
