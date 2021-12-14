from .lossfactory import LossRepro, LossInit
from ...dataset import flipSMPLPoses, flipPoint2D
from .optimize_base import _optimizeSMPL, deepcopy_tensor, dict_of_tensor_to_numpy
import torch
import numpy as np


def flipSMPLPosesV(params,real_id,src_id):
    nViews = params['poses'].shape[0]

    for i in range(nViews):
        if real_id == src_id:
            if i !=src_id:
                params['poses'][i] = flipSMPLPoses(params['poses'][real_id])
        else:
            if i==real_id:
                params['poses'][i] = flipSMPLPoses(params['poses'][src_id])
            else:
                params['poses'][i] = params['poses'][src_id]
    return params

class LossKeypointsMirror2D(LossRepro):
    def __init__(self, keypoints2d, bboxes, Pall, cfg) -> None:
        super().__init__(bboxes, keypoints2d, cfg)
        self.Pall = torch.Tensor(Pall).to(cfg.device)
        self.nJoints = keypoints2d.shape[-2]
        self.nViews = self.keypoints2d.shape[0]
        self.kpt_homo = torch.ones((keypoints2d.shape[0], keypoints2d.shape[1], 1), device=cfg.device)
        print('kpts_homo: {}'.format(self.kpt_homo.shape)) ##delete here
        print('self.keypoints2d: {}'.format(self.keypoints2d.shape)) ## delete here
        print('Pall: {}'.format(Pall.shape)) # delete here
        self.norm = 'l2'

    def residual(self, kpts_est):
        # kpts_est: (2xnFrames, nJoints, 3)
        print('kpts_est: {}'.format(kpts_est.shape))  ## delete here
        kpts_homo = torch.cat([kpts_est[..., :self.nJoints, :], self.kpt_homo], dim=2)
        point_cam = torch.einsum('ab,fnb->fna', self.Pall, kpts_homo)
        img_points = point_cam[..., :2]/point_cam[..., 2:]
        img_points = img_points.view(self.nViews, self.nJoints, 2)
        print('img_points:{}'.format(img_points.shape))
        print('in residual,self.keypoints2d:{}'.format(self.keypoints2d.shape))
        print('self.conf:{}'.format(self.conf.shape))
        residual = (img_points - self.keypoints2d) * self.conf
        return residual

    def __call__(self, kpts_est, **kwargs):
        "reprojection error for mirror"
        # kpts_est: (2xnFrames, 25, 3)
        #kpts_est = kpts_est.reshape()
        kpts_homo = torch.cat([kpts_est[..., :self.nJoints, :], self.kpt_homo], dim=2)
        point_cam = torch.einsum('ab,fnb->fna', self.Pall, kpts_homo)
        img_points = point_cam[..., :2]/point_cam[..., 2:]
        img_points = img_points.view(self.nViews, self.nJoints, 2)
        return super().__call__(img_points)/self.nViews

    def __str__(self) -> str:
        return 'Loss function for Reprojection error of Mirror'


class LossMirrorSymmetry:
    def __init__(self, real_id, N_JOINTS=25, normal=None, cfg=None) -> None:
        """
        real_id: the id of the real view
        """
        idx0, idx1 = np.meshgrid(np.arange(N_JOINTS), np.arange(N_JOINTS))
        idx0, idx1 = idx0.reshape(-1), idx1.reshape(-1)
        idx_diff = np.where(idx0!=idx1)[0]
        self.idx00, self.idx11 = idx0[idx_diff], idx1[idx_diff]
        self.N_JOINTS = N_JOINTS
        self.idx0 = idx0
        self.idx1 = idx1
        self.real_id = real_id
        if normal is not None:
            self.normal = torch.Tensor(normal).to(cfg.device)
            self.normal = self.normal.expand(-1, N_JOINTS, -1)
        else:
            self.normal = None
        self.device = cfg.device

    def parallel_self(self,kpts_est,**kwargs):
        """ encourage parallel to self """
        # kpts_est: (nViews,nJoints,3)
        kpts_real = flipPoint2D(kpts_est[self.real_id,...])
        nViews = kpts_est.shape[0]
        loss = 0
        print('in parallel_self, nViews = {}'.format(nViews))
        for i in range(nViews):
            direct = kpts_est[i,...] - kpts_real
            #print('direct: {}'.format(direct))
            #direct_norm = direct/torch.norm(direct,dim=-1,keepdim=True) ## keepdim=True?
            direct_norm = torch.nn.functional.normalize(direct,dim = -1)
            #print('direct_norm: {}'.format(direct_norm))
            loss += torch.sum(torch.norm(torch.cross(direct_norm[self.idx0,:],direct_norm[self.idx1,:]),dim=1))/self.idx0.shape[0]
        return loss/nViews

    def vertical_self(self,kpts_est,**kwargs):
        """ encourage vertical to self"""
        # kpts_est: (nViews,nJoints,3)
        kpts_real = flipPoint2D(kpts_est[self.real_id,...])
        nViews = kpts_est.shape[0]
        loss = 0

        for i in range(nViews):
            if i != self.real_id:
                direct = kpts_est[i,...] - kpts_real
                direct_norm = torch.nn.functional.normalize(direct,dim = -1)

                mid_point = (kpts_real + kpts_est[i,...])/2

                inner = torch.abs(torch.sum((mid_point[self.idx00, :] - mid_point[self.idx11, :])*direct_norm[self.idx11, :], dim=1))
                loss += torch.sum(inner)/self.idx00.shape[0]
        
        return loss/nViews

    def __str__(self) -> str:
            return 'Loss function for Mirror Symmetry'


def viewSelection(params,body_model,loss_repro,real_id):
    res = []

    nViews = params['poses'].shape[0]
    for i in range(nViews):
        params_new = flipSMPLPosesV(params,real_id,i)
        kpts_est = body_model(return_verts=False, return_tensor=True, **params_new)
        print('kpts_est before residual:{}'.format(kpts_est.shape))
        residual = loss_repro.residual(kpts_est)
        print('residual:{}'.format(residual.shape)) ## delete here
        res_new = torch.norm(residual,dim=-1).mean(dim=-1).sum(dim=0)
        print('res_new:{}'.format(res_new)) ## delete here
        res_new = res_new.item()
        res.append(res_new)
    
    print('res: {}'.format(len(res))) ## delete here
    id_dst = np.argmin(np.array(res))
    
    params_output = flipSMPLPosesV(params,real_id,id_dst)

    return params_output

def optimizeMirrorSoft(body_model, params, bboxes, keypoints2d,Pall,normal,weight,cfg, real_id):
    """
        simple function for optimizing mirror
        Args:
            body_model (smpl)
            params (DictParam): poses(nViews,72), shapes(1,10), Rh(2,3), Th(2,3)
            bboxes (nViews,5) 
            keypoints2d (nViews, nJoints, 3): 2d keypoints of each view
            weight (Dict): string:float
            cfg (Config): config node controling running mode
    """
    nViews = keypoints2d.shape[0]
    prepare_funcs = [
        deepcopy_tensor,
        flipSMPLPosesV,  ## Note here   
    ]

    loss_sym = LossMirrorSymmetry(real_id=real_id, normal=normal, cfg=cfg)
    loss_repro = LossKeypointsMirror2D(keypoints2d, bboxes, Pall, cfg)
    params = viewSelection(params, body_model, loss_repro,real_id)
    init = LossInit(params, cfg)
    loss_funcs = {
        'k2d': loss_repro.__call__,
        'init_poses': init.init_poses,
        'init_shapes': init.init_shapes,
        'par_self': loss_sym.parallel_self,
        'ver_self': loss_sym.vertical_self,
    } 
    postprocess_funcs = [
        dict_of_tensor_to_numpy,
        flipSMPLPosesV  ## Note here! 
    ]
    params = _optimizeSMPL(body_model, params, prepare_funcs, postprocess_funcs, loss_funcs, real_id=real_id, weight_loss=weight, cfg=cfg)
    return params