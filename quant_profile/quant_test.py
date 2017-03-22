import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import pickle
import numpy as np
from tensorflow.python.platform import gfile

testing_file = './test.p'

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_test, y_test = test['features'], test['labels']
n_classes = np.amax(test['labels']) + 1

BATCH_SIZE = 20


# restore graph check example here: 
# https://www.tensorflow.org/versions/r0.10/how_tos/meta_graph/
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)



#correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
#accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run("accu:0", feed_dict={"input_x:0": batch_x, "input_y:0": batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


sess = tf.Session()
#import the graph
with gfile.FastGFile("./quantized_frozen_model.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
#read the weights,bias,etc
#saver.restore(sess, tf.train.latest_checkpoint('./'))
# set as default since the evaluate function need to read default session
with sess.as_default():
#    for i in range(0, 100):
    test_accuracy = evaluate(X_test, y_test)

print("Test Accuracy = {:.3f}".format(test_accuracy))