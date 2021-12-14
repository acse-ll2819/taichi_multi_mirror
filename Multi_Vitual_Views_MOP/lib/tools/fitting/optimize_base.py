import numpy as np
import torch
from .lbfgs import LBFGS
from .lossfactory import LossInit
from .optimize import FittingMonitor, grad_require


def dict_of_tensor_to_numpy(body_params):
    body_params = {key:val.detach().cpu().numpy() for key, val in body_params.items()}
    return body_params

def get_optParams(body_params, cfg, extra_params):
    for key, val in body_params.items():
        body_params[key] = torch.Tensor(val).to(cfg.device)
    if cfg is None:
        opt_params = [body_params['Rh'], body_params['Th'], body_params['poses']]
    else:
        if extra_params is not None:
            opt_params = extra_params
        else:
            opt_params = []
        if cfg.OPT_R:
            opt_params.append(body_params['Rh'])
        if cfg.OPT_T:
            opt_params.append(body_params['Th'])
        if cfg.OPT_POSE:
            opt_params.append(body_params['poses'])
        if cfg.OPT_SHAPE:
            opt_params.append(body_params['shapes'])
        if cfg.OPT_EXPR and cfg.model == 'smplx':
            opt_params.append(body_params['expression'])
    return opt_params

def deepcopy_tensor(body_params):
    for key in body_params.keys():
        body_params[key] = body_params[key].clone()
    return body_params

def _optimizeSMPL(body_model, body_params, prepare_funcs, postprocess_funcs, 
    loss_funcs, real_id, extra_params=None,
    weight_loss={}, cfg=None):
    """ A common interface for different optimization.
    Args:
        body_model (SMPL model)
        body_params (DictParam): poses(1, 72), shapes(1, 10), Rh(1, 3), Th(1, 3)
        prepare_funcs (List): functions for prepare
        loss_funcs (Dict): functions for loss
        weight_loss (Dict): weight
        cfg (Config): Config Node controling running mode
    """
    loss_funcs = {key: val for key, val in loss_funcs.items() if key in weight_loss.keys() and weight_loss[key] > 0.}
    if cfg.verbose:
        print('Loss Functions: ')
        for key, func in loss_funcs.items():
            print('  -> {:15s}: {}'.format(key, func.__doc__))
    opt_params = get_optParams(body_params, cfg, extra_params)
    grad_require(opt_params, True)
    optimizer = LBFGS(opt_params, 
        line_search_fn='strong_wolfe')
    PRINT_STEP = 100
    records = []
    def closure(debug=False):
        # 0. Prepare body parameters => new_params
        optimizer.zero_grad()
        new_params = body_params.copy()
        #for func in prepare_funcs:
        #    new_params = func(new_params)
        func1 = prepare_funcs[0] # deepcopytensor
        func2 = prepare_funcs[1] # flipsmplvpose
        new_params = func1(new_params)
        new_params = func2(new_params, real_id=real_id, src_id=real_id)
        
        # 1. Compute keypoints => kpts_est
        kpts_est = body_model(return_verts=False, return_tensor=True, **new_params)
        #print('in closure, kpts_est:{} '.format(kpts_est))
        # 2. Compute loss => loss_dict
        loss_dict = {key:func(kpts_est=kpts_est, **new_params) for key, func in loss_funcs.items()}
        # 3. Summary and log
        cnt = len(records)
        if cfg.verbose and cnt % PRINT_STEP == 0:
            print('{:-6d}: '.format(cnt) + ' '.join([key + ' %f'%(loss_dict[key].item()*weight_loss[key]) 
                for key in loss_dict.keys() if weight_loss[key]>0]))
        loss = sum([loss_dict[key]*weight_loss[key]
                    for key in loss_dict.keys()])
        for key in loss_dict.keys():
            print('key: {}, loss = {}'.format(key, loss_dict[key]))
        print('loss = {}'.format(loss))
        records.append(loss.item())
        if debug:
            return loss_dict
        loss.backward()
        return loss

    fitting = FittingMonitor(ftol=1e-4)
    final_loss = fitting.run_fitting(optimizer, closure, opt_params)
    fitting.close()
    grad_require(opt_params, False)
    loss_dict = closure(debug=True)
    if cfg.verbose:
        print('{:-6d}: '.format(len(records)) + ' '.join([key + ' %f'%(loss_dict[key].item()*weight_loss[key]) 
            for key in loss_dict.keys() if weight_loss[key]>0]))
    loss_dict = {key:val.item() for key, val in loss_dict.items()}
    # post-process the body_parameters
    #for func in postprocess_funcs:
    #    body_params = func(body_params)
    func3 = postprocess_funcs[0]
    func4 = postprocess_funcs[1]
    body_params = func3(body_params)
    body_params = func4(body_params,real_id=real_id, src_id=real_id)
    return body_params