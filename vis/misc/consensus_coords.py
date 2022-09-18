import sys
import numpy as np                      # for data storage & manipulation
import pickle
import pandas as pd                     # for data parsing & manipulation
import math
import os, cv2, glob
import argparse
import pandas as pd
from collections import Counter
from sklearn.cluster import DBSCAN

db_dir = "/home/n10203478/EGH400/database/"
p2p_dir = "/home/n10203478/koaladetection/CrowdCounting-P2PNet/"  
out_scatterimg = '../consensus_5/'     # name of dir to save scatter plot images to
no_runs = 5

if not os.path.exists(out_scatterimg):
    os.mkdir(out_scatterimg)

# store ground truth for test set in test_array
with open(os.path.join(db_dir,"test_gt.txt")) as load_file:
    gt_test = [tuple(line.split()) for line in load_file]

test_array=[]
for file in glob.glob(os.path.join(db_dir,"test/*.png")): 
    img = cv2.imread(file)
    filename = os.path.basename(file)#.split(".")[0]    
    pre = filename.split(".")[0]    

    for line in gt_test:
        if line[0] == filename:
            count = line[1]
            test_array.append(tuple([pre,count,img]))   

def cluster_img(filename, eps=6.6, min_samples = 3): #outputs filename, no.clusters, mean co-ords

    pts = [] 
    X = [] 

    # Obtain pred points from all 5 trains and store in array X
    for ii in range(no_runs): 
        pts_path = (os.path.join(p2p_dir,"vis","run_"+str(ii+1),filename+'.txt'))
        with open(pts_path) as load_file:
            pts.append([(line.split()) for line in load_file])

    for ii in range(no_runs): 
        for p in pts[ii]:
            x = int(float(p[0])*640)
            y = int(float(p[1])*512)
            X.append([x,y])
    X = np.array(X)

    # Return pred count of 0 if X is empty & skip clustering
    if X.size < 1:
        K=0
        return []

    else:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = clustering.labels_ # measure the performance of dbscan algo
        K = len(set(labels)) - (1 if -1 in labels else 0) # number of clusters

        # Find Cluster means 
        means = np.array([X[labels == i].mean(axis=0) for i in range(0, K)])
        means = means.astype(int)

    return means

    # return[filename, K, means]

def parse_p2p(filepth, coords, median_w=21, median_h=21, img_w=640, img_h=512): 
    obj_id = 0 #deer
    arbitary_w = median_w/img_w #  median width = 21
    arbitary_h = median_h/img_h #  median height = 21

    with open(filepth, "w") as f:
        for lines in coords:
            print(lines)
            # line =  (lines.tostring().split())
            plot_x = float(lines[0])/img_w # x center - arbitary_w/2
            plot_y = float(lines[1])/img_h # y center - arbitary_h/2
            p = "{} {} {} {} {}\n"
            f.write(p.format(obj_id,plot_x,plot_y,arbitary_w,arbitary_h))

def main():
    eps = 6.6
    min_samples = 3

    for line in gt_test:
        filename = (line[0])[:-4]
        count = int(line[1])

        means = cluster_img(filename, eps, min_samples)
        # print("next:", means)

        filepth = os.path.join(out_scatterimg,filename+".txt")

        if len(means)>0:
            parse_p2p(filepth, means)
        else:
            open(filepth,'w').close()

if __name__ == '__main__':
    main()