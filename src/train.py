# -*- coding: utf-8 -*-

"""Implementation of Mobilenet V3.
Architecture: https://arxiv.org/pdf/1905.02244.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
from model import mobilenet_v3_small , mobilenet_v3_large
import src.dataloader as dataloader
from src.utils import normalize, train_in_batch, shuffle_data, augument_data

__all__ = ['mobilenet_v3_large', 'mobilenet_v3_small']


def accuracy(prediction, labels):
        #print("pred:",prediction)
        #print("labels: ",labels)
        labels = tf.cast(tf.reshape(labels,[-1]),tf.int64)
        prediction = tf.argmax(prediction, 1)
        correct_prediction = tf.equal(prediction, labels)
        #print("correct: ",correct_prediction)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #print("acc: ",accuracy)
        return accuracy

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def semi_triplet_loss(Embedding, labels):
    labels = tf.reshape(labels,[-1])
    prediction_semi = tf.nn.l2_normalize(Embedding,axis=1)
    loss_semi = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels,prediction_semi)
    return loss_semi

def cross_entropy(prediction, labels):
    labels = tf.reshape(labels,[-1])
    sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=prediction)
    sparse = tf.reduce_mean(sparse)
    return sparse



if __name__ == "__main__":
    ''' Original Testing '''
    print("begin ...")
    input_test = tf.zeros([8, 224, 224, 3])
    num_classes = 103
    # model, end_points = mobilenet_v3_large(input_test, num_classes, multiplier=1.0, is_training=True, reuse=None)
    # model, end_points = mobilenet_v3_small(input_test, num_classes, multiplier=1.0, is_training=True, reuse=None)
    print("done !")

    ''' Application '''
    data_path = 'dataset/downloads'
    tf.reset_default_graph()
    with tf.name_scope('input'):
        inputs = tf.placeholder(tf.float32, [None, 224, 224, 1])        
        y_true = tf.placeholder(tf.int32, [None, 1])
        alpha = tf.placeholder(tf.float32)
        beta = tf.placeholder(tf.float32)
    with tf.name_scope('stem'):
            
        # out_triplet,out_softmax = MobileV3(inputs)
        softmax, triplet, end_points = mobilenet_v3_small(
            inputs, # input_test
            num_classes,
            multiplier=1.0, 
            is_training=True, 
            reuse=None
        ) # model: softmax outcome
            
    #print(softmax)
    #print(y_true)
    with tf.name_scope('loss'):
        # alpha = 1 / ( (batch_size*(batch_size-1)) * (batch_size**2 - batch_size) )
        # triplet
        triplet_loss = semi_triplet_loss(triplet,y_true)
        # softmax
        softmax_loss = cross_entropy(softmax, y_true)

        loss = tf.add(tf.multiply(alpha,triplet_loss), tf.multiply(beta, softmax_loss))        
        #optim = tf.train.AdamOptimizer().minimize(loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  
        with tf.control_dependencies(update_ops): #保证train_op在update_ops执行之后再执行。  
            optim = tf.train.AdamOptimizer().minimize(loss) 

        # exp step size
        # global_step = tf.Variable(0, trainable=False)
        # starter_learning_rate = 0.05
        # learning_rate = tf.train.exponential_decay(starter_learning_rate,
        #                                       global_step,
        #                                       100, 0.9, staircase=True)
        # optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        acc=accuracy(softmax, y_true)

    epochs = 3000
    batch_size_triplet = 4
    batch_size_softmax = 64
    iteration = 10000//batch_size_softmax
    #saver = tf.train.Saver()
    saver = tf.train.Saver(var_list=tf.global_variables())
    max_val_acc = 0
    start_triplet = 0
    min_triplet_loss = 1
    triplet_goal = 0.1
    logfile = open("log.txt", "w")


    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, "./mobile_checkpoint/new_mobile_val.ckpt")
        triplet_TrainData, TestData = dataloader.dataloader(data_path,batch_size_triplet)
        softmax_TrainData = dataloader.triplet_to_softmax(triplet_TrainData)
        x_test,y_test,end=dataloader.GenNoBatch(TestData)
        print("Start Training")
        for epoch in range(epochs):
            ########Training Stage#######
            if start_triplet == 0:
                if max_val_acc > 0.7:
                    start_triplet = 1
            if start_triplet == 1:
                if min_triplet_loss < triplet_goal:
                    start_triplet = 2

            #######Which Type of Data#######
            if start_triplet == 0:
                TrainData=copy.deepcopy(softmax_TrainData)
            else:
                TrainData=copy.deepcopy(triplet_TrainData)


            training_loss = 0 
            training_acc = 0
            triplet_sum = 0
            real_iteration = 0
            while(True): 
                if start_triplet == 1:
                    alpha_input = 1
                    beta_input = 0
                    x_train,y_train,end=dataloader.GenBatch(TrainData,batch_size_triplet) 
                elif start_triplet == 2:
                    alpha_input = 0.5
                    beta_input = 0.5
                    x_train,y_train,end=dataloader.GenBatch(TrainData,batch_size_triplet) 
                else:
                    alpha_input = 0
                    beta_input = 1
                    x_train,y_train=dataloader.GenRandomBatch(TrainData,batch_size_softmax)

                #######Stop Condition#######
                if start_triplet == 0:
                    if real_iteration >= iteration:
                        break
                else:
                    if end == 1:
                        break

                real_iteration += 1


                x_train = augument_data(x_train,False)
                training_loss_batch, _,training_acc_batch,pred,TL,SL = sess.run(
                    [loss,optim,acc,softmax,triplet_loss,softmax_loss],
                    feed_dict={inputs:x_train,y_true:y_train,alpha:alpha_input,beta:beta_input}
                )
                training_acc += training_acc_batch
                triplet_sum += TL
            
            testing_acc = sess.run(
                [acc],
                feed_dict={inputs:x_test,y_true:y_test,alpha:alpha_input,beta:beta_input}
            )
            #######Saving Condition#######
            if start_triplet == 2 and (triplet_sum / real_iteration) < min_triplet_loss and testing_acc > max_val_acc:
                saver.save(sess, "./mobile_checkpoint/new_mobile_goal.ckpt")
                
            if testing_acc[0] > max_val_acc:
                max_val_acc = testing_acc[0]
            if triplet_sum / real_iteration < min_triplet_loss:
                min_triplet_loss = triplet_sum / real_iteration

            print("epoch:%d Stage:%d train_acc:%.3f val_acc:%.3f triplet_loss:%.3f max_val_acc:%.3f min_TL:%.3f"
                %(epoch, start_triplet, training_acc/real_iteration, testing_acc[0], 
                  triplet_sum/real_iteration, max_val_acc, min_triplet_loss)
            )
            logfile.write("epoch:%d Stage:%d train_acc:%.3f val_acc:%.3f triplet_loss:%.3f max_val_acc:%.3f min_TL:%.3f\n"
                %(epoch, start_triplet, training_acc/real_iteration, testing_acc[0], 
                  triplet_sum/real_iteration, max_val_acc, min_triplet_loss)
            )

        saver.save(sess, "./mobile_checkpoint/new_mobile_val_last.ckpt")
   
