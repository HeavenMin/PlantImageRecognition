"""
    Min Gao
    773090
"""
import os
import imghdr

for (dirpath, dirnames, filenames) in os.walk("/Users/heaven/Desktop/GRADUATION_PROJECT/flowerImages"):
    for filename in filenames:
        if imghdr.what(dirpath + '/' +filename) != 'jpeg':
            print(filename)
            os.remove(dirpath + '/' + filename)
