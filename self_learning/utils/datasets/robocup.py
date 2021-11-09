import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data

__all__ = ['RobocupDataset']

#assume robocup dataset contains a file that lists all the image names.
class RobocupDataset(data.Dataset):
    def __init__(self, dataset, root, transforms=None):
        self.dataset = dataset
        self.transforms = transforms
        self.ims,  = self.parse_location_txt()
        self.data_dir = os.path.join(root, dataset)
        self.num = len(self.ims)

    def __getitem__(self, index):
        """Return:
           dict:'im' is the image tensor
                'xyz' is the absolute position of the image
                'wpqr' is the  absolute rotation quaternion of the image
        """
        data_dict = {}
        im = self.ims[index]
        data_dict['im_ref'] = im
        im = Image.open(os.path.join(self.data_dir, im))
        if self.transforms:
            im = self.transforms(im)
        data_dict['im'] = im
        #data_dict['xyz'] = self.poses[index][0]
        #data_dict['wpqr'] = self.poses[index][1]
        return data_dict

    def __len__(self):
        return self.num

    def parse_location_txt(self, fpath):
        '''
        get all the file addresses
        '''
        #poses = []
        ims = []
        f = open(fpath)
        for line in f.readlines()[3::]:
            cur = line.strip().split(' ')
            #xyz = np.array([float(v) for v in cur[1:4]], dtype=np.float32)
            #wpqr = np.array([float(v) for v in cur[4:8]], dtype=np.float32)
            ims.append(cur[0])
            #poses.append((xyz, wpqr))
        f.close()
        return ims

    def __repr__(self):
        fmt_str = 'AbsPoseDataset {}\n'.format(self.dataset)
        fmt_str += 'Number of samples: {}\n'.format(self.__len__())
        fmt_str += 'Root location: {}\n'.format(self.data_dir)
        fmt_str += 'Pose txt: {}\n'.format(self.pose_txt)
        fmt_str += 'Transforms: {}\n'.format(self.transforms.__repr__().replace('\n', '\n    '))
        return fmt_str

