import numpy as np
import cv2
import os
import json
from os.path import join
from ..tools.writer import FileWriter
def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

class ImageData:
    """
    data for image: including camera data of the image, bounding boxes,image,keypoints 2d
    """

    def __init__(self,image_path, annot_root, out_root, kpts_type='body25',config={}) -> None:
        head, tail = os.path.split(image_path)
        if 'jpg' in tail:
            self.base = tail.replace('.jpg','')
        if 'png' in tail:
            self.base = tail.replace('.png','')
        base_op = self.base + '.json'
        op_annot_path = join(annot_root,base_op)
        self.op_full_path = op_annot_path
        self.img = cv2.imread(image_path)
        self.height = 0
        self.width = 0
        self.annots = []
        self.out_root = out_root
        self.writer = FileWriter(self.out_root, config=config)

    def read_annots(self):
        # convert the keypoints 2d list to numpy

        data = read_json(self.op_full_path)
        annots = data['annots']

        for i,annot in enumerate(annots):
            kpts_old = annot['keypoints']
            for j in range(len(kpts_old)):
                kpts_old[j] = np.array(kpts_old[j])
            kpts_new = np.vstack(kpts_old)
            annots[i]['keypoints'] = kpts_new
        
        return annots

    def load_annots(self):
        """
        load bounding box, keypoints 2d
        """
        #print(self.op_full_path)
        assert os.path.exists(self.op_full_path)
        data = read_json(self.op_full_path)

        self.annots = self.read_annots()
        self.height = data['height']
        self.width = data['width']

    def load_camera(self):
        assert os.path.exists(self.op_full_path)
        data = read_json(self.op_full_path)
        if 'K' not in data.keys():
            height, width = self.height, self.width
            focal = 1.2*min(height,width) # as colmap
            K = np.array([focal, 0., width/2, 0., focal, height/2, 0. ,0., 1.]).reshape(3, 3)
        else:
            K = np.array(data['K']).reshape(3, 3)

        camera = {'K':K ,'R': np.eye(3), 'T': np.zeros((3, 1)), 'dist': np.zeros((1, 5))}
        print('in dataset, camera R: {}'.format(camera['R']))
        print('in dataset, camera T: {}'.format(camera['T']))
        camera['RT'] = np.hstack((camera['R'], camera['T']))
        camera['P'] = camera['K'] @ np.hstack((camera['R'], camera['T']))
        print('in dataset, camera P: {}'.format(camera['P']))
        return camera

    def write_keypoints3d(self, keypoints3d):
        outname = join(self.out_root, 'keypoints3d','{}.json'.format(self.base))
        save_json(outname,keypoints3d)
    
    def vis_smpl(self,render_data, image, camera):
        outname = join(self.out_root,'smpl','{}.jpg'.format(self.base))
        images = [image]
        for key in camera.keys():
            camera[key] = camera[key][None,...]
            self.writer.vis_smpl(render_data,images,camera,outname,add_back=True)