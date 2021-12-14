import numpy as np
import os
from tqdm import tqdm
import torch
import json


def grad_require(paras, flag=False):
    if isinstance(paras, list):
        for par in paras:
            par.requires_grad = flag 
    elif isinstance(paras, dict):
        for key, par in paras.items():
            par.requires_grad = flag



def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])

class FittingMonitor:
    def __init__(self, ftol=1e-5, gtol=1e-6, maxiters=100, visualize=False, verbose=False, **kwargs):
        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol
        self.visualize = visualize
        self.verbose = verbose
        if self.visualize:
            from utils.mesh_viewer import MeshViewer
            self.mv = MeshViewer(width=1024, height=1024, bg_color=[1.0, 1.0, 1.0, 1.0], 
                body_color=[0.65098039, 0.74117647, 0.85882353, 1.0],
            offscreen=False)

    def run_fitting(self, optimizer, closure, params, smpl_render=None, **kwargs):
        prev_loss = None
        grad_require(params, True)
        if self.verbose:
            trange = tqdm(range(self.maxiters), desc='Fitting')
        else:
            trange = range(self.maxiters)
        for iter in trange:
            loss = optimizer.step(closure)
            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break
            
            # if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
            #         for var in params if var.grad is not None]):
            #     print('Small grad, stopping!')                
            #     break

            if iter > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break
            
            if self.visualize:
                vertices = smpl_render.GetVertices(**kwargs)
                self.mv.update_mesh(vertices[::10], smpl_render.faces)
            prev_loss = loss.item()
        grad_require(params, False)        
        return prev_loss

    def close(self):
        if self.visualize:
            self.mv.close_viewer()