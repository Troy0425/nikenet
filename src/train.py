# -*- coding: utf-8 -*-

"""Implementation of Mobilenet V3.
Architecture: https://arxiv.org/pdf/1905.02244.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import dataloader
from utils import normalize, train_in_batch, shuffle_data, augument_data
import tensorflow as tf
import copy
import numpy as np
import random
from model import mobilenet_v3_small , mobilenet_v3_large

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

def accuracy(prediction, labels):
	labels = tf.cast(tf.reshape(labels,[-1]),tf.int64)
	prediction = tf.argmax(prediction, 1)
	correct_prediction = tf.equal(prediction, labels)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accuracy
def triplet_val(ground_truth_vector, ground_truth_label, testing_vector, testing_label):
    correct = 0
    number = len(testing_vector)
    for i in range(len(testing_vector)):
        dis_index = [None]*103
        index = [i for i in range(103)]
        for j in range(len(ground_truth_vector)):
            d = np.linalg.norm(testing_vector[i] - ground_truth_vector[j])
            #dis_index.append([d,j])
            label_index=ground_truth_label[j][0]
            if(dis_index[label_index]!=None):
                if(dis_index[label_index]>d):
                    dis_index[label_index]=d
            else:
                dis_index[label_index]=d
        top_index = sorted(zip(dis_index, index))[:1]    
        for idx in top_index:
            if testing_label[i] == idx[1]:
                correct += 1
                break
    return correct / number
if __name__ == "__main__":
    ''' Original Testing '''
    print("begin ...")
    input_test = tf.zeros([8, 224, 224, 3])
    num_classes = 103
    # model, end_points = mobilenet_v3_large(input_test, num_classes, multiplier=1.0, is_training=True, reuse=None)
    # model, end_points = mobilenet_v3_small(input_test, num_classes, multiplier=1.0, is_training=True, reuse=None)
    print("done !")

    ''' Application '''
    data_path = 'dataset/image103'
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
        adam = tf.train.AdamOptimizer()
        optim = adam.minimize(loss)
        #trainer = tf.compat.v1.train.AdamOptimizer() 
        #optim = trainer.minimize(loss)
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
    saver = tf.train.Saver()
    max_val_acc = 0
    start_triplet = 0
    max_val_triplet_acc = 0
    triplet_goal = 0.7
    softmax_goal = 0.7
    val_triplet_acc = 0
    logfile = open("log.txt", "w")

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, "./mobile_checkpoint/320.ckpt")
        triplet_TrainData, TestData = dataloader.dataloader(data_path)
        softmax_TrainData = dataloader.triplet_to_softmax(triplet_TrainData)
        x_test,y_test,end=dataloader.GenNoBatch(TestData)
        print("Start Training")
        #######Prepare triplet validtion data####### 
        train_val_data = []
        train_val_label = []
        for i in range(len(triplet_TrainData)):
            sample_index = random.sample(range(0, len(triplet_TrainData[i])), 10) 
            for j in sample_index:
                train_val_data.append(dataloader.loaddata(triplet_TrainData[i][j][0]))
                train_val_label.append([triplet_TrainData[i][j][1]])
        for epoch in range(epochs):
            ########Training Stage#######
            if start_triplet == 0:
                if max_val_acc > softmax_goal:
                    start_triplet = 1
            if start_triplet == 1:
                if max_val_triplet_acc > triplet_goal:
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
                x_train = x_train/255.0

                training_loss_batch, _,training_acc_batch,pred,TL,SL = sess.run(
                    [loss,optim,acc,softmax,triplet_loss,softmax_loss],

                    feed_dict={inputs:x_train,y_true:y_train,alpha:alpha_input,beta:beta_input}
                )
                training_acc += training_acc_batch
                triplet_sum += TL
                #print(opt)
            #######Testing softmax#######
            testing_acc, test_val_vector,lr= sess.run(
                [acc,triplet,adam._lr_t],
                feed_dict={inputs:x_test,y_true:y_test,alpha:alpha_input,beta:beta_input}
            )
            ######Testing triplet########
            if start_triplet != 0:
                train_val_vector = sess.run([triplet],feed_dict={inputs:train_val_data, y_true: train_val_label})
                val_triplet_acc = triplet_val(train_val_vector[0], train_val_label, test_val_vector, y_test)
            #######Saving Condition#######
            if start_triplet != 0 and val_triplet_acc > max_val_triplet_acc :
                print("Save!!!")
                #saver.save(sess, "./mobile_checkpoint/%d.ckpt"%(epoch))

                
            if testing_acc > max_val_acc:
                max_val_acc = testing_acc
            if val_triplet_acc > max_val_triplet_acc:
                max_val_triplet_acc = val_triplet_acc

            print("epoch:%d Stage:%d train_acc:%.3f val_acc:%.3f triplet_val_acc:%.3f max_val_acc:%.3f max_val_triplet_acc:%.3f lr:%.6f"
                %(epoch, start_triplet, training_acc/real_iteration, testing_acc, 
                  val_triplet_acc, max_val_acc, max_val_triplet_acc,lr)
            )
            logfile.write("epoch:%d Stage:%d train_acc:%.3f val_acc:%.3f triplet_val_acc:%.3f max_val_acc:%.3f max_val_triplet_acc:%.3f lr:%.6f\n"
                %(epoch, start_triplet, training_acc/real_iteration, testing_acc, 
                  val_triplet_acc, max_val_acc, max_val_triplet_acc,lr)
            )

        saver.save(sess, "./mobile_checkpoint/new_mobile_val_last.ckpt")
   
