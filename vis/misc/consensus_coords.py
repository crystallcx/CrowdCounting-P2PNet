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

def get_clus_means(p2p_dir, no_runs, filename, eps=6.6, min_samples = 3): #outputs filename, no.clusters, mean co-ords

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

def parse_p2p(filepth, coords, median_w=21, median_h=21, img_w=640, img_h=512): 
    obj_id = 0 #deer
    arbitary_w = median_w/img_w #  median width = 21
    arbitary_h = median_h/img_h #  median height = 21

    with open(filepth, "w") as f:
        for lines in coords:
            print(lines)
            # line =  (lines.tostring().split())
            plot_x = float(lines[0])/img_w # x center 
            plot_y = float(lines[1])/img_h # y center 
            p = "{} {}\n"
            # f.write(p.format(obj_id,plot_x,plot_y,arbitary_w,arbitary_h))  #yolov5 format
            f.write(p.format(plot_x, plot_y))  #co-ord format


def main():
    # setup command line arguments
    parser = argparse.ArgumentParser(description='Consensus Coordinates')

    parser.add_argument("--db_dir", action="store", dest="db_dir", default="/home/n10203478/EGH400/database/") # database directory
    parser.add_argument("--p2p_dir", action="store", dest="p2p_dir", default="/home/n10203478/koaladetection/CrowdCounting-P2PNet/") # p2pnet directory
    parser.add_argument("--out_dir", action="store", dest="out_pth", default='../consensus/' )  # name of dir to save scatter plot images to
    parser.add_argument("--no_runs", type=int, dest="no_runs", default=None) # number of runs
    parser.add_argument("--eps", type=float, dest="eps", default=None) # eps value for DBSCAN
    parser.add_argument("--minpts", type=int, dest="minpts", default=None) # minimum pts value for DBSCAN

    args = parser.parse_args()  

    if not os.path.exists(args.out_pth):
        os.mkdir(args.out_pth)

    # store ground truth for test set in test_array
    with open(os.path.join(args.db_dir,"test_gt.txt")) as load_file:
        gt_test = [tuple(line.split()) for line in load_file]

    # for 5 runs, 6.6, 3
    # for 10 runs, 5.5, 5

    for line in gt_test:
        filename = (line[0])[:-4]
        means = get_clus_means(args.p2p_dir, args.no_runs, filename, args.eps, args.minpts)

        filepth = os.path.join(args.out_pth,filename+".txt")
        if len(means) > 0:
            parse_p2p(filepth, means)
        else:
            open(filepth,'w').close()

if __name__ == '__main__':
    main()