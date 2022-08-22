import sys
import numpy as np                      # for data storage & manipulation
import pickle
import pandas as pd                     # for data parsing & manipulation
import math
import os, cv2, glob
import argparse
import pandas as pd
import tensorflow as tf             # for deep learning
from matplotlib import pyplot as plt    # for plotting
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

db_dir = "/home/n10203478/EGH400/database/"
p2p_dir = "/home/n10203478/EGH400/koaladetection/CrowdCounting-P2PNet" 
p2ptr_names=['run_1','run_2', 'run_3', 'run_4', 'run_5take2']

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

# def get_args_parser():
#     parser = argparse.ArgumentParser('Set params', add_help=False)
#     parser.add_argument('--p2p_dir', default="/home/n10203478/EGH400/koaladetection/CrowdCounting-P2PNet" ,
#                         help='path where p2p repo is located')
#     parser.add_argument('--database_dir', default="/home/n10203478/EGH400/database/",
#                         help='path where image database is located')
#     return parser

def column(matrix, i):
    return [row[i] for row in matrix]

def cluster_img(k, eps=5, min_samples = 3): #outputs filename, no.clusters, mean co-ords

    pts = [] 
    X = [] 

    filename = test_array[k][0]+".png"

    # Obtain pred points from all 5 trains and store in array X
    for i in range(len(p2ptr_names)): 
        pts_path = (os.path.join(p2p_dir,"vis",p2ptr_names[i],test_array[k][0]+'.txt'))
        with open(pts_path) as load_file:
            pts.append([(line.split()) for line in load_file])

    for i in range(len(p2ptr_names)): 
        for p in pts[i]:
            x = int(float(p[0])*640)
            y = int(float(p[1])*512)
            X.append([x,y])
    X = np.array(X)

    # Return pred count of 0 if X is empty & skip clustering
    if X.size < 1:
        K=0
        return[filename, 0]

    else:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = clustering.labels_ # measure the performance of dbscan algo
        K = len(set(labels)) - (1 if -1 in labels else 0) # number of clusters

        # Find Cluster means 
        means = np.array([X[labels == i].mean(axis=0) for i in range(0, K)])
        means = means.astype(int)

    return[filename, K, means]


def cluster_all(eps, min_samples,length):
    cluster_points=[]
    for m in range(length):
        cluster_points.append(cluster_img(m, eps, min_samples))
    return cluster_points

gt_pts_test = []
for line in gt_test:
    filename = (line[0])
    count = int(line[1])
    gt_pts_test.append([filename, count])

gt_test_sorted = sorted(gt_pts_test, key=lambda tup: tup[0])
test_x = column(gt_test_sorted, 1)

# eps_arr = [1, 2.5, 5, 7.5, 10]
# min_pts_arr = [1, 2, 3, 4, 5]
# eps_arr = np.linspace(5,10,30) #[1, 2,3,4,5,6,7,8,9, 10]
eps_arr = np.linspace(1,10,45) #[1, 2,3,4,5,6,7,8,9, 10]

# eps_arr = [x * 0.2 for x in range(1, 10)]
min_pts_arr = range(1,len(p2ptr_names))#[1, 2, 3, 4, 5]

output = []
# smallest = 1
for eps in eps_arr:
    eps = round(eps,1)
    for min_pts in min_pts_arr:
        cluster_pts = cluster_all(eps, min_pts,len(gt_test)) 

        cluster_pts_sorted = sorted(cluster_pts, key=lambda tup: tup[0])
        test_y = column(cluster_pts_sorted, 1) 

        correct = 0
        for i in range(len(test_y)):
            if test_y[i] == test_x[i]:
                correct += 1

        acc = round(correct/len(gt_test),5)

        mae = round(metrics.mean_absolute_error(test_x,test_y),5)
        mse = round(metrics.mean_squared_error(test_x,test_y),5)

        # if mae < smallest:
        #     smallest = mae
        #     output.append([eps, min_pts, mse, mae, acc])

        if mae < .4:
            output.append([eps, min_pts, mse, mae, acc])

with open('clus_scan_out.txt', 'w') as f:
    f.write("eps min_pts  mse  mae  accuracy\n")
    for line in output:
        f.write(f"{line}\n")
