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
p2p_dir = "/home/n10203478/koaladetection/CrowdCounting-P2PNet/"  
out_scatterimg = './scatterimages_5adasda/'     # name of dir to save scatter plot images to
out_txtname = 'clus_scan_out_1asda0_test.txt'  # name of txt file to print results to
# p2ptr_names=['run_1','run_2', 'run_3', 'run_4', 'run_5']  # expected naming convention
no_runs = 5


if not os.path.exists(out_scatterimg):
    os.mkdir(out_scatterimg)

# store ground truth for test set in test_array
with open(os.path.join(db_dir,"test_gt.txt")) as load_file:
    gt_test = [tuple(line.split()) for line in load_file]


def column(matrix, i):
    return [row[i] for row in matrix]

def cluster_img(p2p_dir, no_runs, filename, eps=6.6, min_samples = 3): #outputs filename, no.clusters, mean co-ords
    pts = [] 
    X = [] 

    # Obtain pred points from all 5 trains and store in array X
    for ii in range(no_runs): 
        pts_path = (os.path.join(p2p_dir,"vis","run_"+str(ii+1),filename+'.txt'))
        # pts_path = (os.path.join(p2p_dir,"vis",p2ptr_names[i],test_array[k][0]+'.txt'))
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

    return [K, means]

def dbscan_all(gt_test, out_txtname):
    # DBSCAN
    eps_arr = np.arange(1, 10, 0.5).tolist()
    min_pts_arr = np.arange(3, 10, 1).tolist()

    output = []
    for eps in eps_arr:
        eps = round(eps,1)
        for min_pts in min_pts_arr:

            cluster_pts = []
            for line in gt_test:
                filename = jd[0]
                cluster_pts.append(p2pdir, no_runs, filename, eps, min_pts) 

            cluster_pts_sorted = sorted(cluster_pts, key=lambda tup: tup[0])
            test_y = column(cluster_pts_sorted, 1) 

            correct = 0
            for ii in range(len(test_y)):
                if test_y[ii] == test_x[ii]:
                    correct += 1

            acc = round(correct/len(gt_test),5)

            mae = round(metrics.mean_absolute_error(test_x,test_y),5)
            mse = round(metrics.mean_squared_error(test_x,test_y),5)

            # only record value if below 0.3
            if mse<= 0.5 or mae <=0.3:
                output.append([eps, min_pts, mse, mae, acc])

            if img:
                generate_img(test_x,eps,min_pts,cluster_pts)
                
    with open(out_txtname, 'w') as f:
    # f.write(" ".join(p2ptr_names) + "\n")
    f.write("eps min_pts  mse  mae  accuracy\n")
    for line in output:
        f.write(f"{line}\n")


def generate_img(test_x,eps,min_pts,cluster_pts ):
        cluster_pts_sorted = sorted(cluster_pts, key=lambda tup: tup[0])
        test_y=column(cluster_pts_sorted, 1) 

        #Generate a list of unique points
        points=list(set(zip(test_x,test_y))) 
        #Generate a list of point counts
        count=[len([x for x,y in zip(test_x,test_y) if x==p[0] and y==p[1]]) for p in points]

        # Plotting:
        plot_x=[i[0] for i in points]
        plot_y=[i[1] for i in points]
        count=np.array(count)

        fig=plt.figure(figsize=[8,4])
        ax=fig.add_subplot(1, 1, 1)
        plt.xlim(-0.5, max(test_x)+1)
        plt.ylim(-0.75, 14)
        plt.scatter(plot_x,plot_y,c=count,s=30*count**0.5,cmap='Spectral_r', linewidths=0.4, edgecolors=(0,0,0))
        plt.colorbar()
        plt.xlabel('Ground Truth Count\nEps:{}, Min pts={}\nmae: {}, mse: {}'.format(eps,min_pts,mae,mse)), plt.ylabel('Predicted Count')
        plt.title("Test set: Cluster Consensus vs Ground Truth Scatterplot")
        plt.grid()

        out_pth = os.path.join(out_scatterimg, 'scatter_e{}_m{}.png'.format(eps,min_pts) )
        fig.savefig(out_pth,bbox_inches='tight', dpi=100)
        plt.close('all')

def main():
    # setup command line arguments
    parser.add_argument("--db_dir", action="store", dest="db_dir", default="/home/n10203478/EGH400/database/") # database directory
    parser.add_argument("--p2p_dir", action="store", dest="p2p_dir", default="/home/n10203478/koaladetection/CrowdCounting-P2PNet/") # p2pnet directory
    parser.add_argument("--outpath", action="store", dest="out_pth", default='../consensus_10/' )  # name of dir to save scatter plot images to
    parser.add_argument("--no_runs", type=int, dest="no_runs", default=None) # number of runs
    parser.add_argument("--eps", type=float, dest="eps", default=None) # eps value for DBSCAN
    parser.add_argument("--minpts", type=int, dest="minpts", default=None) # minimum pts value for DBSCAN

    args = parser.parse_args()  

    gt_pts_test = []
    for line in gt_test:
        filename = (line[0])
        count = int(line[1])
        gt_pts_test.append([filename, count])

    # sort gt_test by count
    gt_test_sorted = sorted(gt_pts_test, key=lambda tup: tup[0])
    test_x = column(gt_test_sorted, 1)



    # -------------------------------------------------------------------------
    # Generating scatterplot of cluster results (for visualisation) - comment out section if not needed.
    # -------------------------------------------------------------------------
    # if out_scatterimg:
       


if __name__ == '__main__':
    main()