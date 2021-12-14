import os
import json
import numpy as np
import cv2
from os.path import join
from .vis_base import plot_bbox, plot_keypoints, merge

mkdir = lambda x:os.makedirs(x, exist_ok=True)
mkout = lambda x:mkdir(os.path.dirname(x))

class FileWriter:
    def __init__(self, output_path, config=None, basenames=[], cfg=None) -> None:
        self.out = output_path
        self.basenames = basenames

        self.save_origin = False
        self.config = config

    def vis_smpl(self, render_data, images, cameras, outname, add_back):
        mkout(outname)
        from ..visualize.renderer import Renderer
        render = Renderer(height=1024, width=1024, faces=None)
        render_results = render.render(render_data, cameras,images, add_back=add_back)
        image_vis = merge(render_results, resize=not self.save_origin)
        cv2.imwrite(outname, image_vis)
        return image_vis