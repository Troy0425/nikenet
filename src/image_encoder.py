import warnings
warnings.filterwarnings('ignore')

import cv2
import tensorflow as tf
import numpy as np

from src.train import mobilenet_v3_small

def encode_standard_array(img_path):

    im = cv2.imread(img_path)
    # print(im.shape)
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)
    # print(im.shape)
    img_array = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # print(x_train.shape)
    img_array=img_array[:,:,np.newaxis]
    img_array=img_array[np.newaxis,:,:,:]/255.0

    return img_array

def embed_shoe(img_array, model_path="./model/350.ckpt"):

#     img_array = encode_standard_array(path)
    tf.reset_default_graph()
    
    with tf.name_scope('input'):
            inputs = tf.placeholder(tf.float32, [None, 224, 224, 1]) ##輸入為四維[Batch_size,height,width,channels] 
    with tf.name_scope('stem'):

        # out_triplet,out_softmax = MobileV3(inputs)
        softmax, triplet, end_points = mobilenet_v3_small(
            inputs, # input_test
            103,
            multiplier=1.0, 
            is_training=True, 
            reuse=None
        ) # model: softmax outcome
        
    saver = tf.train.Saver()

    with tf.Session() as sess:
            
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path) 
        pred = sess.run(
                        [triplet], 
                        feed_dict={inputs:img_array}) #list

        pred = pred[0]    

    return pred

if __name__ == "__main__":

    path = "image/airforce.jpg"
    
    img_array = encode_standard_array(path)
    emb_vec = embed_shoe(img_array)

    print(emb_vec.shape)
    # print(emb_vec)
    # np.save("nike_classifier/image/test_array.npy", emb_vec)
