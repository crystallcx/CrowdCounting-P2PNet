import os, cv2
import numpy
import argparse
import glob

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for visualising P2Pnet trains', add_help=False)

    parser.add_argument('--output_dir', default='../../../CrowdCounting-P2PNet/vis/test_all',  #<--------------------------
                        help='path where to save')
    parser.add_argument('--input_dir', default='../../../database/test/',
                        help='path of database test set')
    parser.add_argument('--num_trains', default=5,
                        help='path where the trained weights saved')
    return parser

# assumes there are folders within .../CrowdCounting-P2PNet/vis named run_1, run_2, etc
# which contain p2p point predictions in .txt files for each image 
# <filename> must be identical to names in database/test/*.png
# vis
# | run_1
#   | <filename>.txt
#   | ...
# | run_2
#   | <filename>.txt>
#   | ...
# ...

def main(args):

    db_dir = os.path.dirname(os.path.normpath(args.input_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for file in glob.glob(os.path.join(args.input_dir,"*.png")):
        img = cv2.imread(file)
        pre = os.path.basename(file).split(".")[0]    
        colors = [[0, 255, 0], [255, 0, 0],[0, 255, 255],[255, 0, 255],[255, 255, 0]]

        gt_pth = os.path.join(args.input_dir,pre+'.txt')

        pts = []
        img_to_draw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(args.num_trains): 
            pts_path = (os.path.join(os.path.dirname(args.output_dir),"run_"+str(i+1),pre+'.txt'))

            with open(pts_path) as load_file:
                pts.append([(line.split()) for line in load_file])

        with open(gt_pth) as load_file:
            gt_points = [(line.split()) for line in load_file]
            
        for p in gt_points:
            x = int(float(p[1])*640)
            y = int(float(p[2])*512)
            img_to_draw=cv2.circle(img_to_draw, (x,y), 2, (0, 0, 255), -1)

        for i in range(args.num_trains): 
            for p in pts[i]:
                x = int(float(p[0])*640)
                y = int(float(p[1])*512)
                img_to_draw=cv2.circle(img_to_draw, (x, y), 1, colors[i], -1)

        cv2.imwrite(os.path.join(args.output_dir, '{}_gt{}.jpg'.format(pre,len(gt_points))), img_to_draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot prediction points from multiple trained P2Pmodels', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)