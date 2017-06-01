#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : Lang
 PURPOSE : download flower image from Flickr and Googel then fliter fake jpeg
 Environment : This code can only run in python3 and need flickrapi and icrawler
 Usage : python3 "this_file_name" "aim_file_with_flower_name_info" "download_images_path"
"""

__author__ = 'Lang'

import re
import os
import sys
import json
import urllib.request
import shutil
import imghdr
from icrawler.builtin import GoogleImageCrawler
from flickrapi import FlickrAPI

flowerNameSetFile_path = sys.argv[1]
image_path = sys.argv[2]
image = urllib.request.URLopener()
FLICKR_PUBLIC = '367aed513876077c1cdcadb29d88ef02'
FLICKR_SECRET = '9b6e223653519900'
flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
google_count = 0
flickr_count = 0

#Record the process of downloading in file "record.txt"
def record(source,flowerID):
    if source == "Google":
        global google_count
        google_count = google_count +1
        record = open("record.txt",'a')
        record.write("Google process finished flower: " + flowerID + ", " + str(google_count) + " kinds of Google flower finished.\n")
        record.close()
    else:
        if source == "Flickr":
            global flickr_count
            flickr_count = flickr_count +1
            record = open("record.txt",'a')
            record.write("Flickr process finished flower: " + flowerID + ", " + str(flickr_count) + " kinds of Flickr flower finished.\n")
            record.close()
        else:
            record = open("record.txt",'a')
            record.write("all finished.\n")
            record.close()

#Get Flower's name from file contains all flowers' name, each line present one flower's name.
def getFlowerName(path):
    flowerNameList = []
    with open(path, "r", encoding="UTF-8") as f:
        for flowerName in f:
            flowerNameList.append(flowerName[:-1])
    return flowerNameList

#Download images from google according to flowers' name and create not existed directories.
def downloadFlowersGoogle(flowerList ,image_path):
    for name in flowerList:
        imageDir = image_path + '/' + name
        try:
            os.stat(imageDir)
        except:
            os.mkdir(imageDir)
        google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                            storage={'root_dir' : imageDir})
        google_crawler.crawl(keyword= name, max_num=1000,
                             date_min=None, date_max=None,
                             min_size=(200,200), max_size=None)
        record("Google",name)

#Download images from flickr according to flowers' name.
def downloadFlowersFlickr(flowerList ,image_path):
    max_photo = 2000
    extras='url_sq,url_t,url_s,url_q,url_m,url_n,url_z,url_c,url_l,url_o'
    for name in flowerList:
        imageDir = image_path + '/' + name
        downloaded_photo = 0
        page_number = 0
        while downloaded_photo <= max_photo:
            plants = flickr.photos.search(text="\"" + name + "\"", per_page=500, extras=extras,page = page_number)
            photos = plants['photos']
            for i in range(0,len(photos['photo'])):
                try:
                    url = photos['photo'][i]['url_c']
                    downloaded_photo = downloaded_photo + 1
                    image.retrieve(url,imageDir + "/" + str(downloaded_photo)+".jpg")
                except:
                    try:
                        url = photos['photo'][i]['url_z']
                        downloaded_photo = downloaded_photo + 1
                        image.retrieve(url,imageDir + "/" + str(downloaded_photo)+".jpg")
                    except:
                        try:
                            url = photos['photo'][i]['url_n']
                            downloaded_photo = downloaded_photo + 1
                            image.retrieve(url,imageDir + "/" + str(downloaded_photo)+".jpg")
                        except:
                            try:
                                url = photos['photo'][i]['url_m']
                                downloaded_photo = downloaded_photo + 1
                                image.retrieve(url,imageDir + "/" + str(downloaded_photo)+".jpg")
                            except:
                                try:
                                    url = photos['photo'][i]['url_q']
                                    downloaded_photo = downloaded_photo + 1
                                    image.retrieve(url,imageDir + "/" + str(downloaded_photo)+".jpg")
                                except:
                                    print("no picture for " + name + " photo NO." + str(downloaded_photo))
            if len(photos['photo']) == 500:
                page_number = page_number + 1
            else:
                break
        record("Flickr",name)

#Delete not jpeg image part1.
def delNotJpeg(path):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if not filename.endswith('.jpg'):
                os.remove(dirpath + '/' + filename)
        for dirname in dirnames:
            if dirname.startswith('0'):
                shutil.rmtree(dirpath + '/' + dirname)

#Delete not jpeg image part2.
def delFakeJpeg(path):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if imghdr.what(dirpath + '/' +filename) != 'jpeg':
                os.remove(dirpath + '/' + filename)

#Rename all images downloaded for future processing.
def rename(path):
    flowerDirList = sorted(os.listdir(path))
    for flowerID in flowerDirList:
        imageDir = path + '/' + flowerID
        images = sorted(os.listdir(imageDir))
        count = 0
        for image in images:
            if ".jpg" in image:
                count = count + 1
                os.rename(imageDir + '/' + image, imageDir + '/' + flowerID + " " + str(count) + ".jpg")


if __name__ == '__main__':
    flowerList = sorted(getFlowerName(flowerNameSetFile_path))
    downloadFlowersGoogle(flowerList, image_path)
    downloadFlowersFlickr(flowerList, image_path)
    delNotJpeg(image_path)
    delFakeJpeg(image_path)
    rename(image_path)
    record("finished","All Finished")
