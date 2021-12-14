import os, sys
import cv2
from os.path import join
from tqdm import tqdm
from glob import glob
import numpy as np

mkdir = lambda x: os.makedirs(x, exist_ok=True)


def extract_2d(openpose, image, keypoints, render, args):
    skip = False
    if os.path.exists(keypoints):
        # check the number of images and keypoints
        if len(os.listdir(image)) == len(os.listdir(keypoints)):
            skip = True
    if not skip:
        os.makedirs(keypoints, exist_ok=True)
        if os.name != 'nt':
            cmd = './build/examples/openpose/openpose.bin --image_dir {} --write_json {} --display 0'.format(image, keypoints)
        else:
            cmd = 'bin\\OpenPoseDemo.exe --image_dir {} --write_json {} --display 0'.format(join(os.getcwd(),image), join(os.getcwd(),keypoints))
        if args.highres!=1:
            cmd = cmd + ' --net_resolution -1x{}'.format(int(16*((368*args.highres)//16)))
        if args.handface:
            cmd = cmd + ' --hand --face'
        if args.render:
            if os.path.exists(join(os.getcwd(),render)):
                cmd = cmd + ' --write_images {}'.format(join(os.getcwd(),render))
            else:
                os.makedirs(join(os.getcwd(),render), exist_ok=True)
                cmd = cmd + ' --write_images {}'.format(join(os.getcwd(),render))
        else:
            cmd = cmd + ' --render_pose 0'
        os.chdir(openpose)
        os.system(cmd)

import json
def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def bbox_from_openpose(keypoints, rescale=1.2, detection_thresh=0.01):
    """Get center and scale for bounding box from openpose detections."""
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)
    # adjust bounding box tightness
    bbox_size = bbox_size * rescale
    bbox = [
        center[0] - bbox_size[0]/2, 
        center[1] - bbox_size[1]/2,
        center[0] + bbox_size[0]/2, 
        center[1] + bbox_size[1]/2,
        keypoints[valid, 2].mean()
    ]
    return bbox

def load_openpose(opname):
    mapname = {'face_keypoints_2d':'face2d', 'hand_left_keypoints_2d':'handl2d', 'hand_right_keypoints_2d':'handr2d'}
    assert os.path.exists(opname), opname
    data = read_json(opname)
    out = []
    pid = 0
    for i, d in enumerate(data['people']):
        keypoints = d['pose_keypoints_2d']
        keypoints = np.array(keypoints).reshape(-1, 3)
        annot = {
            'bbox': bbox_from_openpose(keypoints),
            'personID': pid + i,
            'keypoints': keypoints.tolist(),
            'isKeyframe': False
        }
        for key in ['face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            if len(d[key]) == 0:
                continue
            kpts = np.array(d[key]).reshape(-1, 3)
            annot[mapname[key]] = kpts.tolist()
        out.append(annot)
    return out


def create_annot_file(imgname):
    assert os.path.exists(imgname), imgname
    img = cv2.imread(imgname)
    height, width = img.shape[0], img.shape[1]
    imgnamesep = imgname.split(os.sep)
    #filename = os.sep.join(imgnamesep[imgnamesep.index('images'):])
    annot = {
        'filename':imgname,
        'height':height,
        'width':width,
        'annots': [],
        'isKeyframe': False
    }
    #save_json(annotname, annot)
    return annot

def convert_from_openpose(img_dir,annot_dir):
    # convert the 2d pose from openpose
    os.chdir(img_dir)
    inputlist = sorted(os.listdir(img_dir))

    for inp in tqdm(inputlist,desc='{:10s}'.format(os.path.basename(annot_dir))):
        if '.jpg' in inp:
            head, tail = os.path.split(inp)
            base = tail.replace('.jpg','')
            base_op = base + '_keypoints.json'
            op_annot_path = join('openpose',base_op) 
            annots = load_openpose(op_annot_path)
            annotname = join(annot_dir,base+'.json')
            annot = create_annot_file(inp)
            annot['annots'] = annots
            save_json(annotname,annot)
        if '.png' in inp:
            head, tail = os.path.split(inp)
            base = tail.replace('.png','')
            base_op = base + '_keypoints.json'
            op_annot_path = join('openpose',base_op) 
            annots = load_openpose(op_annot_path)
            annotname = join(annot_dir,base+'.json')
            annot = create_annot_file(inp)
            annot['annots'] = annots
            save_json(annotname,annot)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="the path of data")
    parser.add_argument('--ext', type=str, default='jpg', choices=['jpg', 'png'], help="image file extension")
    parser.add_argument('--annot', type=str, default='annots', help="sub directory name to store the generated annotation files, default to be annots")
    parser.add_argument('--handface', action='store_true')
    parser.add_argument('--openpose', type=str, 
        default='../../openpose')
    parser.add_argument('--render', action='store_true', 
        help='use to render the openpose 2d')
    parser.add_argument('--path_origin', default=os.getcwd())
    parser.add_argument('--highres', type=float, default=1)
    args = parser.parse_args()

    path_origin = os.getcwd()
    if os.path.isdir(args.path):
        
        image_root = args.path
        annot_root = args.annot
        if os.path.exists(annot_root):
            extract_2d(args.openpose, image_root, 
                        join(args.path, 'openpose'), 
                        join(args.path, 'openpose_render'), args)
            os.chdir(path_origin)
            convert_from_openpose(
                        image_root,annot_root
                    )
    else:
        print(args.path, ' not exists')