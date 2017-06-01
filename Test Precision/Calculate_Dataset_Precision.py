
"""
 AUTHOR : Lang
 PURPOSE : Multi Self Deep Learning
"""

__author__ = 'Lang'

import tensorflow as tf, sys
import os
import shutil


# change this as you see fit

graph_path_temple = sys.argv[1]
label_path_temple = sys.argv[2]

graph_path = os.path.abspath(graph_path_temple)
label_path = os.path.abspath(label_path_temple)

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile(label_path)]


# Unpersists graph from file
with tf.gfile.FastGFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


tracing = open("precision.txt",'w')
tracing.close()

directory = os.listdir('.')
sorted_directory = sorted(directory)

for image_dir_path in sorted_directory:
    try:
        image_number = 0
        image_precision = 0
        image_predicit_right = 0
        for image_path in sorted(os.listdir(image_dir_path)):
            try:
                # Read in the image_data
                image_data = tf.gfile.FastGFile(image_dir_path+'/'+image_path, 'rb').read()
                with tf.Session() as sess:
                    # Feed the image_data as input to the graph and get first prediction
                    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

                    predictions = sess.run(softmax_tensor, \
                        {'DecodeJpeg/contents:0': image_data})

                    # Sort to show labels of first prediction in order of confidence
                    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                    node_count = 0
                    for node_id in top_k:
                        node_count = node_count + 1
                        human_string = label_lines[node_id]
                        score = predictions[0][node_id]
                        if node_count == 1 and image_dir_path.lower() == human_string.lower():
                            image_number = image_number + 1
                            image_predicit_right = image_predicit_right + 1
                            image_precision = image_precision + score
                            print("image " + image_path + " processing complete,preidict right")
                        if node_count != 1 and image_dir_path.lower() == human_string.lower():
                            image_number = image_number + 1
                            image_precision = image_precision + score
                            print("image " + image_path + " processing complete,predict wrong")
            except:
                print(image_dir_path + " " + image_path + " processing went wrong.")
        tracing = open("precision.txt",'a')
        average_precision = image_precision / image_number
        right_predicit = image_predicit_right / image_number
        tracing.write("There are " + str(image_number) + " images in folder " + image_dir_path + ", " + str(right_predicit) + " percentage number predicted right and " + str(average_precision) + " is the average prediction where the image was predicted as its belong category.\n")
        tracing.close()
    except:
        print('error:'+ image_dir_path)

tracing = open("precision.txt",'a')
tracing.write("all finished.")
tracing.close()
