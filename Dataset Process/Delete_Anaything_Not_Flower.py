
"""
 AUTHOR : Lang
 PURPOSE : Multi Self Deep Learning
"""

__author__ = 'Lang'

import tensorflow as tf, sys
import os


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

count = 0

tracing = open("processing.txt",'w')
tracing.close()

for image_dir_path in os.listdir('.'):
    try:
        for image_path in os.listdir(image_dir_path):
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
                    if label_lines[top_k[0]] == "no":
                        os.remove(image_dir_path+'/'+image_path)
                        print('removed picture '+image_path)
                    else:
                        print('remain picture '+image_path)
            except:
                os.remove(image_dir_path+'/'+image_path)
                print('removed picture'+image_path)
        count = count +1
        tracing = open("processing.txt",'a')
        tracing.write("finish " + str(count) + " kinds of removing not flower pictures\n")
        tracing.close()
    except:
        print('error:'+ image_dir_path)

tracing = open("processing.txt",'a')
tracing.write("all finished")
tracing.close()
