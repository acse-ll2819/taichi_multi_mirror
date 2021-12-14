from operator import imod
import numpy as np
from tqdm import tqdm
from os.path import join
from operator import imod
import numpy as np
from tqdm import tqdm
from os.path import join
import os
import cv2

from lib.models.smplmodel import load_model, merge_params, select_nf
from lib.dataset import ImageData
from lib.models.estimator import SPIN, init_with_spin
from lib.tools import simple_assign
from lib.pipeline.mirror import multi_stage_optimize

import json
def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data



def demo(image_path,annot_root,out_root,body_model,spin_model,args):
    """
    optimize single image
    path: the path of this single image
    body_model: smpl is the default model
    """

    dataset = ImageData(image_path,annot_root,out_root)
    dataset.load_annots()
    camera = dataset.load_camera()
    annots = dataset.annots
    #image = dataset.img
    image = cv2.imread(image_path)
    # initialize the smpl parameters
    body_params_all = []
    bboxes,keypoints2d,pids = [],[],[]
    for i,annot in enumerate(annots):
        assert annot['personID'] == i, (i, annot['personID'])
        result = init_with_spin(body_model,spin_model,image,annot['bbox'],annot['keypoints'],camera)
        body_params_all.append(result['body_params'])
        bboxes.append(annot['bbox'])
        keypoints2d.append(annot['keypoints'])
        pids.append(annot['personID'])

    bboxes = np.vstack(bboxes)
    keypoints2d = np.stack(keypoints2d)
    body_params = merge_params(body_params_all)
    
    keypoints3d = body_model(return_verts=False,return_tensor=False,**body_params)
    # get the id of the real view
    real_id = simple_assign(keypoints3d)
 

    ######## test ###############
    print('body_params[poses]: {}'.format(body_params['poses'].shape))
    print('body_params[shapes]:{}'.format(body_params['shapes'].shape))
    print('bboxes: {}'.format(bboxes.shape))
    print('keypoints2d: {}'.format(keypoints2d.shape))
    
    ########### end ##############
    normal = None ## we do not use mirror normal constraint
    body_params = multi_stage_optimize(body_model, body_params, bboxes, keypoints2d, Pall=camera['P'], normal=normal, real_id=real_id, args=args)
    vertices = body_model(return_verts=True, return_tensor=False, **body_params)
    keypoints = body_model(return_verts=False, return_tensor=False, **body_params)

    ###### test ###########
    print('output vertices: {}'.format(vertices.shape))
    print('output keypoints: {}'.format(keypoints.shape))
    ####### end ##########
    # write output data
    write_data = [{'id': pids[i], 'keypoints3d': keypoints[i]} for i in range(len(pids))]
    if os.path.isdir(join(out_root, 'keypoints3d'))==False:
        os.makedirs(join(out_root, 'keypoints3d'))
    if os.path.isdir(join(out_root, 'smpl'))==False:
        os.makedirs(join(out_root, 'smpl'))
    #dataset.write_keypoints3d(write_data)

    if args.vis_smpl:
        render_data = {pids[i]: {
                'vertices': vertices[i], 
                'faces': body_model.faces, 
                'vid': 0, 'name': 'human_{}'.format(pids[i])} for i in range(len(pids))}
        dataset.vis_smpl(render_data, image,camera)
    ####### test ###########
    print('vertices: {}'.format(vertices.shape))
    print('keypoints: {}'.format(keypoints.shape))
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="the path of images")
    parser.add_argument('annot_path', type=str, help="the path of annotation folder")
    parser.add_argument('output_path', type=str, help="the path of output folder")
    parser.add_argument('--model', type=str, default='smpl', help="type of body model")
    parser.add_argument('--gender', type=str, default='neutral', help="the path of annotation folder")
    parser.add_argument('--vis_smpl', action='store_true', help='set to visualize the smpl result')
    args = parser.parse_args()

    #with Timer('Loading {}, {}'.format(args.model, args.gender)):
    body_model = load_model(args.gender, model_type=args.model)
    #Timer('Loading SPIN'):
    spin_model = SPIN(
            SMPL_MEAN_PARAMS='data/models/smpl_mean_params.npz', 
            checkpoint='data/models/spin_checkpoint.pt', 
            device=body_model.device)
    
    inputlist = sorted(os.listdir(args.path))

    for inp in inputlist:
        if '.jpg' in inp:
            demo(join(args.path,inp),args.annot_path,args.output_path,body_model,spin_model,args)
        if '.png' in inp:
            demo(join(args.path,inp),args.annot_path,args.output_path,body_model,spin_model,args)

