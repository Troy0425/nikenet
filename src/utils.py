import random
from datetime import datetime

import requests
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
#import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from google_images_download import google_images_download 

def normalize(X_train,X_test):
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

def shuffle_data(x_train, y_train):
    shuffle= list(zip(x_train, y_train))

    np.random.shuffle(shuffle)

    x_train, y_train= zip(*shuffle)
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    return (x_train,y_train)

def train_in_batch(x_train, y_train,batch_size):
    assert len(x_train)/batch_size==int(len(x_train)/batch_size)
    Batch_X=np.split(x_train, len(x_train)/batch_size)
    Batch_y=np.split(y_train, len(y_train)/batch_size)
    for x,y in zip(Batch_X,Batch_y):
        yield (x,y)

def augument_data(batch_imgs, if_includ_orig=False):
    '''
    activate your virtual "pip install imgaug" before implementation.
    
    Input:
        batch_imgs: batch of (height, weight, channel) 
    Output:
        augmented_imgs: 
    '''
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.SomeOf((0,3),[
        #iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Multiply(),
        iaa.Fliplr(0.5),
        iaa.Flipud(),
         # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0))
         # blur images with a sigma of 0 to 3.0
    ],random_order=True) # can be modified to various type
    aug_imgs = seq.augment_images(batch_imgs)

    if if_includ_orig:
        aug_imgs = np.concate((batch_imgs, aug_imgs), axis=0)    

    return aug_imgs
    """
def plot_loss(loss_dict, epoch_idx, iter_num):
    '''
    Input:
        loss_list: y (length: len(epoch_idx) * iter_nu,)
        epoch_idx: x 
        iter_num: iteration num for each epoch
    '''
    
    train = loss_dict.get('train') 
    test = loss_dict.get('test')
    step_idx = 1/iter_num
    epoch_idx = np.arange(epoch_idx[0], epoch_idx[-1]+1, step_idx)
    
    if test:
        plt.plot(epoch_idx , train.get('loss'),'o-', label='Training Loss')
        plt.plot(epoch_idx , train.get('acc'),'o-', label='Training Accuracy')
        plt.plot(epoch_idx , test.get('loss'),'o-', label='Testing Loss')
        plt.plot(epoch_idx , test.get('acc'),'o-', label='Testing Accuracy')
        plt.legend()
    else:
        plt.plot(epoch_idx , train.get('loss'),'o-', label="Loss")
        plt.plot(epoch_idx , train.get('acc'),'o-', label="Accuracy")
        plt.legend()
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.style.use("fivethirtyeight")
    plt.show()

    now = datetime.now()
    plt.savefig('training_rec/Record_Plot_{}_{}_{}_{}.png'.format(now.strftime("%Y%m%d"), 
                                                                   now.strftime("%H:%M:%S"),
                                                                   epoch_idx[0], epoch_idx[-1]))
"""
def crawl_img_url(keyword):
    '''
    Input : keyword in string format
    Output: url in string format
    '''
    print('Crawling {} ...'.format(keyword))
    keyword = keyword.lower()
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords"     : keyword,
             "limit"        : 5,
             "print_urls"   : True,
             "size"         : "medium", # Possible values: large, medium, icon, >400*300, >640*480, >800*600, >1024*768, >2MP, >4MP, >6MP, >8MP, >10MP, >12MP, >15MP, >20MP, >40MP, >70MP
             "no_download": True
             }
    img_info = response.download(arguments)[0]# .get(keyword) # tuple of info
    for name, img_path_list in img_info.items():
        img_path = random.choice(img_path_list)

    return img_path

def convert_table(table,label):
    all_table=[[] for _ in range(496)]
    class_list=[]
    for i in range(len(table)):
        for vector in table[i]:
            all_table[label[i]].append(vector)
    return all_table

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


if __name__ == "__main__":

    # print(crawl_img_url("air force 1"))
    kw = 'Nike Air Icarus'
    print(crawl_img_url(kw))
    # response = google_images_download.googleimagesdownload()
    # arguments = {"keywords"     : kw,
    #          "limit"        : 5,
    #          "print_urls"   : False,
    #          "size"         : "medium",
    #          "no_download": True
    #          }
    # img_info = response.download(arguments)[0].get(kw.lower()) # tuple of info
    # img_path = random.choice(img_info)
    # print(img_path)

