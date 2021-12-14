import os.path as osp
import torch
import glob
import torch.utils.data as data
from torchvision.io import read_video
import torch.nn.functional as F
import json
import numpy as np
import math


class CATER(data.Dataset):
    MAX_N_OBJ = 10
    RADIUS = 4
    COLORS = torch.ByteTensor([
        [0, 0, 0],
        [0, 0, 128],
        [0, 128, 0],
        [128, 0, 0],
        [0, 0, 255],
        [0, 255, 0]
        [255, 0, 0],
        [255, 255, 255],
        [0, 128, 128],
        [128, 0, 128]
    ])
    
    def __init__(self, args, train=True):
        super().__init__()
        self.train = train
        self.map_scale = args.map_scale
        self.sequence_length = args.sequence_length
        self.hparams = args

        videos = glob.glob(osp.join(args.filelist, 'videos', '*.avi'))
        videos.sort()
        threshold = int(len(videos) * 0.95)
        self.video_paths = videos[:threshold] if train else videos[threshold:]
        self.scene_paths = [osp.join(args.filelist, 'scenes',
                                     osp.basename(path).replace('.avi', '.json'))
                            for path in self.video_paths]
    
    def __len__(self):
        return len(self.scene_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        scene_path = self.scene_paths[idx]

        imgs = read_video(video_path)[0] # THWC in {0, .., 255}
        imgs_orig = imgs[:self.sequence_length]
        H, W = imgs.shape[1:3]
        
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        mean = torch.FloatTensor(mean).view(1, 1, 1, 3)
        std = torch.FloatTensor(std).view(1, 1, 1, 3)
        imgs_orig = ((imgs_orig / 255.) - mean) / std
        
        rsz_h, rsz_w = math.ceil(H / self.map_scale[0]), math.ceil(W /self.map_scale[1])
        object_info = json.load(open(scene_path, 'r'))
        n_objs = len(object_info['objects'])
        labels = torch.zeros((H, W, n_objs), dtype=torch.float32)
        for i, obj in enumerate(object_info['objects']):
            c, r = obj['pixel_coords'][:2]
            cmin, cmax = max(0, c - self.RADIUS), min(W, c + self.RADIUS)
            rmin, rmax = max(0, r - self.RADIUS), min(H, r + self.RADIUS)
            labels[rmin:rmax, cmin:cmax, i] = 1.

        labels_resize = F.interpolate(labels.movedim(-1, 0).unsqueeze(0), size=(rsz_h, rsz_w), mode='bilinear').squeeze(0)
        labels_resize = labels_resize.movedim(0, -1)

        lbl_map = self.COLORS[n_objs].clone()
          
        return imgs, imgs_orig, labels, labels_resize, lbl_map, torch.tensor(0)