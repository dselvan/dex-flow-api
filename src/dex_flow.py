"""
module for predicting classifications of an image against a trained graph
"""
import os
import uuid
import tensorflow as tf


class dex_flow:
    def __init__(self):
        pass

    """method for running tensorflow against a posted image"""

    def identify_pokemon(self, image_data):
        tf.reset_default_graph()

        # Loads label file, strips off carriage return
        label_lines = [line.rstrip() for line
                       in tf.gfile.GFile("retrained_labels.txt")]

        # Unpersists graph from file
        with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        top_id = top_k[0]

        return label_lines[top_id]
