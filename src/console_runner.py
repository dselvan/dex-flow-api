from __future__ import print_function

from dex_flow import dex_flow
import tensorflow as tf


def run():
    df = dex_flow()
    image_data = tf.gfile.FastGFile('bulbasaur.jpg', 'rb').read()
    result = df.identify_pokemon(image_data)
    print(result)


run()
