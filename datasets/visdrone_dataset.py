from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class VisDroneDataset(Dataset):
    """ VisDrone dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        header_names =  ["frame_index","target_id","bbox_left","bbox_top","bbox_width","bbox_height","score","object_category","truncation", "occlusion"]
        self.frames_objects = pd.read_csv(csv_file, header=None, names=header_names)
        self.root_dir = root_dir
        # self.transform = transform

    def __len__(self):
        return len(set(self.frames_objects["frame_index"]))

    def __getitem__(self, index):
        '''
        Args:
            index (int): frame number, start from 0
        returns:
            sample (dict):
                image (skimage): ndarray of frame image
                item (dict): 
                    sequence (string): folder name of the sequence
                    frameID (int): frame number == index
                    gt (list): a list of objects(dict)
                        objects (dict):
                            classID (int): 
                            bbox (list): ["bbox_left","bbox_top","bbox_width","bbox_height"]
                            filename (string): image name. 
                    dim (tuple): (h, w, num_channel)
        '''
        objs = self.frames_objects.loc[self.frames_objects['frame_index'] == index + 1]
        img_name = "{}.jpg".format(str(index + 1).zfill(7))
        item = dict()
        item['sequence'] = self.root_dir.split("/")[-1]
        item['frameID'] = index
        item['gt'] = list()
        for index,row in objs.iterrows():
            obj = dict()
            obj['classID'] = row.iloc[7]
            obj["bbox"] = row.iloc[2:6].values.tolist()
            obj['filename'] = img_name
            item['gt'].append(obj)

        img_path = os.path.join(self.root_dir,
                                img_name)
        image = io.imread(img_path)
        item['dim'] = image.shape
        sample = {'image': image, 'item': item}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample