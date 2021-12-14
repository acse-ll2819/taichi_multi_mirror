import numpy as np
import torch


funcl2 = lambda x: torch.sum(x**2)
funcl1 = lambda x: torch.sum(torch.abs(x**2))

def gmof(squared_res, sigma_squared):
    """
    Geman-McClure error function
    """
    return (sigma_squared * squared_res) / (sigma_squared + squared_res)


class LossRepro:
    def __init__(self, bboxes, keypoints2d, cfg) -> None:
        device = cfg.device
        bbox_sizes = np.maximum(bboxes[..., 2] - bboxes[..., 0], bboxes[..., 3] - bboxes[..., 1])
        # 这里的valid不是一维的，因为不清楚总共有多少维，所以不能遍历去做
        bbox_conf = bboxes[..., 4]
        #bbox_mean_axis = -1
        #bbox_sizes = (bbox_sizes * bbox_conf).sum(axis=bbox_mean_axis)/(1e-3 + bbox_conf.sum(axis=bbox_mean_axis))
        bbox_sizes = bbox_sizes * bbox_conf
        print('in lossrepro, bbox_sizes:{}'.format(bbox_sizes.shape))
        bbox_sizes = bbox_sizes[..., None, None]
        # depress the dead view: set the confidence to 0
        bbox_sizes[bbox_sizes < 10] = 1e6
        inv_bbox_sizes = torch.Tensor(1./bbox_sizes).to(device)
        keypoints2d = torch.Tensor(keypoints2d).to(device)
        self.keypoints2d = keypoints2d[..., :2]

        self.conf = keypoints2d[..., 2:] * inv_bbox_sizes * 100
        print('in lossrepro, conf:{}'.format(self.conf.shape))
        self.norm = 'gm'
    
    def __call__(self, img_points):
        residual = (img_points - self.keypoints2d) * self.conf
        # squared_res: (nFrames, nJoints, 2)
        if self.norm == 'l2':
            squared_res = residual ** 2
        elif self.norm == 'l1':
            squared_res = torch.abs(residual)
        elif self.norm == 'gm':
            squared_res = gmof(residual**2, 200)
        else:
            import ipdb; ipdb.set_trace()
        return torch.sum(squared_res)


class LossInit:
    def __init__(self, params, cfg) -> None:
        self.norm = 'l2'
        self.poses = torch.Tensor(params['poses']).to(cfg.device)
        self.shapes = torch.Tensor(params['shapes']).to(cfg.device)

    def init_poses(self, poses, **kwargs):
        "distance to poses_0"
        if self.norm == 'l2':
            return torch.sum((poses - self.poses)**2)/poses.shape[0]
    
    def init_shapes(self, shapes, **kwargs):
        "distance to shapes_0"
        if self.norm == 'l2':
            return torch.sum((shapes - self.shapes)**2)/shapes.shape[0]