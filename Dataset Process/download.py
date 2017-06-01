
"""
 AUTHOR : Lang
 PURPOSE : Multi Self Deep Learning
"""

__author__ = 'Lang'

import re
import os
import json
import urllib
from flickrapi import FlickrAPI

rf = open('125_small_pictures.txt','rt')
image = urllib.URLopener()


FLICKR_PUBLIC = '367aed513876077c1cdcadb29d88ef02'
FLICKR_SECRET = '9b6e223653519900'
flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
extras='url_sq,url_t,url_s,url_q,url_m,url_n,url_z,url_c,url_l,url_o'
max_photo = 2000
process_count = 0



for line in rf:
    process_count = process_count +1
    split = line.split(',')
    name = split[0].strip()
    sub_name= 0
    count = 0
    page_number = 1
    try:
        os.stat(name)
    except:
        os.mkdir(name)

    while count <= max_photo and sub_name < len(split):
        plants = flickr.photos.search(text="\"" + split[sub_name].strip() + "\"", per_page=500, extras=extras,page = page_number,)
        photos = plants['photos']
        for i in range(0,len(photos['photo'])):
            if count > max_photo:
                break
            try:
                url = photos['photo'][i]['url_sq']
                count = count + 1
                image.retrieve(url,name + "/" + name +" " + str(count)+".jpg")
            except:
                print("no picture for " + name + " photo NO." + str(count))
        if len(photos['photo']) == 500:
            page_number = page_number + 1
        else:
            sub_name = sub_name+1
            page_number = 1
        if count > max_photo:
            break




    print(str(process_count)+' kinds of plants download finished')
