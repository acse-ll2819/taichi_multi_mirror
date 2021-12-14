from .config import Config
from ..tools.fitting import optimizeMirrorSoft

def load_weight_mirror(model):
    if model == 'smpl':
        weight = {
            'k2d': 2e-4, 
            'init_poses': 1e-3, 'init_shapes': 1e-2,
            'smooth_body': 5e-1, 'smooth_poses': 1e-1,
            'par_self': 5e-2, 'ver_self': 2e-2,
            'par_mirror': 5e-2
        }
    return weight


def multi_stage_optimize(body_model, body_params, bboxes, keypoints2d, Pall, normal, real_id, args):
    weight = load_weight_mirror(args.model) ### note argts here
    config = Config()
    config.device = body_model.device
    #config.verbose = args.verbose
    config.OPT_R = True
    config.OPT_T = True
    config.OPT_SHAPE = True

    config.OPT_POSE = False
    body_params = optimizeMirrorSoft(body_model,body_params, bboxes, keypoints2d, Pall,normal,weight,config, real_id)
    config.OPT_POSE = True
    body_params = optimizeMirrorSoft(body_model,body_params, bboxes, keypoints2d, Pall,normal,weight,config, real_id)

    return body_params