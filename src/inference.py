import os
import re
import time
import random
import logging

import numpy as np 
import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity

import src.utils as utils
import src.dataloader as dataloader
import src.image_encoder as image_encoder

class NikeNet():

    def __init__(self):
        logging.basicConfig(level = logging.INFO, 
                            format = '%(levelname)s : %(asctime)s : %(message)s')
        # self.model_type = model_type # self-built, ....
        self.table = np.load("data/496_vec_crop.npy", allow_pickle=True)
        self.label = np.load("data/496_lab_crop.npy", allow_pickle=True)
        self.truth_table= utils.convert_table(self.table,self.label)
        # self.ground_img, self.val_img = dataloader.dataloader("/home/troy0425/image496_nocrop")
        with open("data/ground_img2.txt", "r", encoding='utf-8') as f:
            self.ground_img = [re.sub("\n", "", line) for line in f.readlines()]
        self.img_list = list(np.load('data/img.npy'))

    def _encode_image(self, img_path):
        
        img_array = image_encoder.encode_standard_array(img_path)
        emb_vec = image_encoder.embed_shoe(img_array)

        return emb_vec

    def similarity_class2(self, table, new_emb):
        global_dis=[]
        for class_list in table:
            local_dis=[] 
            for idx in range(len(class_list)):
                local_dis.append(np.linalg.norm(new_emb[-1]-class_list[idx]))
            dis = np.mean(sorted(local_dis)[3:20])
            global_dis.append(dis)
        # global_dis = np.array(global_dis)
        return np.array(global_dis), (np.array(global_dis)).argsort()[:]

    def find_top10(self, img_path):
        # img = []
        # for i in range(100):
        #     img.append(dataloader.loaddata(ground_img[i][0][0]))
        img = self.img_list
        
        # print('For {}'.format(img_path))
        input_img = dataloader.loaddata(img_path)
        img.append(input_img)
        test_emb = image_encoder.embed_shoe(img)

        simi, ans = self.similarity_class2(self.truth_table, test_emb)
        # print(ans)
        cand = [{"name": self.ground_img[index], 
                "similarity": -1*simi[index] } 
                    for index in ans[:6]]
        trans_simi = list(utils.softmax(np.array([info.get('similarity') for info in cand[3:]] ))) # length 6
        
        high_cand = [{"name": cand_info.get('name'), "similarity": str(round(simi,2)    ) } for 
                    cand_info, simi in zip(cand[:3], trans_simi)]
        low_cand = cand[3:]

        for cand_info in high_cand:
            cand_info['img'] = utils.crawl_img_url(cand_info.get('name'))

        cand = {"high": high_cand, "low": low_cand}
        return cand

if __name__ == '__main__':

    nike_net = NikeNet()
    test_img = 'image/airforce.jpg'
    # nike_net.encode_image(test_img)
    top10 = nike_net.find_top10(test_img)
    print(top10.get('high'))
