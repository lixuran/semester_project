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
        self.data_dir = os.path.join(root, dataset)
        self.ims  = self.find_images()
        self.num = len(self.ims)
        self.root = root
        #self.imsl = []
        #for i in range(len(self.ims)):
        #    im = Image.open(os.path.join(self.data_dir,self.ims[i]))
        #    if self.transforms:
        #        im = self.transforms(im)
        #    self.imsl.append(im)
        #print("real data loaded") 

    def __getitem__(self, index):
        """Return:
           dict:'im' is the image tensor
                'xyz' is the absolute position of the image
                'wpqr' is the  absolute rotation quaternion of the image
        """
        data_dict = {}
        im = self.ims[index]
        data_dict['im_ref'] = im
        #im = self.imsl[index]
        im = Image.open(os.path.join(self.data_dir, im))
        if self.transforms:
           im = self.transforms(im)
        data_dict['im'] = im
        #data_dict['xyz'] = self.poses[index][0]
        #data_dict['wpqr'] = self.poses[index][1]
        return data_dict

    def __len__(self):
        return self.num

    def find_images(self):
        '''
        get all the file addresses
        '''
        #poses = []
        #ims = []
	#os.listdir(self.data_dir)
        
        #for line in f.readlines()[3::]:
            #cur = line.strip().split(' ')
            #xyz = np.array([float(v) for v in cur[1:4]], dtype=np.float32)
            #wpqr = np.array([float(v) for v in cur[4:8]], dtype=np.float32)
            #ims.append(cur[0])
            #poses.append((xyz, wpqr))
        #f.close()
        return os.listdir(self.data_dir)

    def __repr__(self):
        fmt_str = 'AbsPoseDataset {}\n'.format(self.dataset)
        fmt_str += 'Number of samples: {}\n'.format(self.__len__())
        fmt_str += 'Root location: {}\n'.format(self.data_dir)
        fmt_str += 'Pose txt: {}\n'.format(self.pose_txt)
        fmt_str += 'Transforms: {}\n'.format(self.transforms.__repr__().replace('\n', '\n    '))
        return fmt_str

