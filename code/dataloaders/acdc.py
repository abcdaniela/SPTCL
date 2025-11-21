import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from torch.utils.data.dataset import Dataset
from random import choice
from utils import *


class ACDC(Dataset):

    def __init__(self, keys, purpose, args):
        self.data_dir = args.data_dir
        self.patch_size = args.patch_size
        self.purpose = purpose
        self.classes = args.classes
        self.do_contrast = args.do_contrast
        self.files = []
        if self.do_contrast:
            # we do not pre-load all data, instead, load data in the get item function
            self.slice_position = []
            self.partition = []
            self.slices = []
            for key in keys:
                frames = subfiles(join(self.data_dir, 'patient_%03d'%key), False, None, ".npy", True)
                for frame in frames:
                    image = np.load(join(self.data_dir, 'patient_%03d'%key, frame))
                    for i in range(image.shape[0]):
                        self.files.append(join(self.data_dir, 'patient_%03d'%key, frame))
                        self.slices.append(i)
                        self.slice_position.append(float(i+1)/image.shape[0])
                        part = image.shape[0] / 4.0
                        if part - int(part) >= 0.5:
                            part = int(part + 1)
                        else:
                            part = int(part)
                        self.partition.append(max(0,min(int(i//part),3)+1))
        else:
            for key in keys:
                frames = subfiles(join(self.data_dir, 'patient_%03d'%key), False, None, ".npy", True)
                for frame in frames:
                    image = np.load(join(self.data_dir, 'patient_%03d'%key, frame))
                    for i in range(image.shape[1]):
                        self.files.append(image[:,i])
        print(f'dataset length: {len(self.files)}')

    def __getitem__(self, index):
        if not self.do_contrast:
            img = self.files[index][0].astype(np.float32)
            label = self.files[index][1]
            img, label = self.prepare_supervised(img, label)
            return img, label
        else:
            img = np.load(self.files[index]).astype(np.float32)[self.slices[index]]
            img1, img2 = self.prepare_contrast(img)
            return img1, img2, self.slice_position[index], self.partition[index]
            
    # this function for normal supervised training
    def prepare_supervised(self, img, label):
        if self.purpose == 'train':
            # resize image
            img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
            label, _  = pad_and_or_crop(label, self.patch_size, mode='fixed', coords=coord)
            # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
            data_dict = {'data':img[None, None], 'seg':label[None, None]}
            tr_transforms = []
            tr_transforms.append(MirrorTransform((0, 1)))
            tr_transforms.append(RndTransform(SpatialTransform(self.patch_size, list(np.array(self.patch_size)//2),
                                                            True, (100., 350.), (14., 17.),
                                                            True, (0, 2.*np.pi), (-0.000001, 0.00001), (-0.000001, 0.00001),
                                                            True, (0.7, 1.3), 'constant', 0, 3, 'constant', 0, 0,
                                                            random_crop=False), prob=0.67, alternative_transform=RandomCropTransform(self.patch_size)))

            train_transform = Compose(tr_transforms)
            data_dict = train_transform(**data_dict)
            img = data_dict.get('data')[0]
            label = data_dict.get('seg')[0]
            return img, label
        else:
            # resize image
            img, coord = pad_and_or_crop(img, self.patch_size, mode='centre')
            label, _  = pad_and_or_crop(label, self.patch_size, mode='fixed', coords=coord)
            return img[None], label[None]

    # use this function for contrastive learning
    def prepare_contrast(self, img):
        # resize image
        img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
        # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
        data_dict = {'data':img[None, None]}
        tr_transforms = []
        tr_transforms.append(MirrorTransform((0, 1)))
        tr_transforms.append(RndTransform(SpatialTransform(self.patch_size, list(np.array(self.patch_size)//2),
                                                        True, (100., 350.), (14., 17.),
                                                        True, (0, 2.*np.pi), (-0.000001, 0.00001), (-0.000001, 0.00001),
                                                        True, (0.7, 1.3), 'constant', 0, 3, 'constant', 0, 0,
                                                        random_crop=False), prob=0.67, alternative_transform=RandomCropTransform(self.patch_size)))

        train_transform = Compose(tr_transforms)
        data_dict1 = train_transform(**data_dict)
        img1 = data_dict1.get('data')[0]
        data_dict2 = train_transform(**data_dict)
        img2 = data_dict2.get('data')[0]
        return img1, img2

    def  __len__(self):
        return len(self.files)


def pad_if_too_small(image, new_shape, pad_value=None):
    """Padding a image according to the new shape.

    The result shape will be [max(image[0], new_shape[0]), max(image[1], new_shape[1])].
    e.g.,
    1. image:[10,20], new_shape:(30,30), the res shape is [30,30].
    2. image:[10,20], new_shape:(10,10), the res shape is [10,20].
    3. image:[3,10,20], new_shape:(3,20,20), the res shape is [3,20,20].

    Args:
      image: a numpy array.
      new_shape: a tuple, # elements should be the same as the image.
      pad_value: padding value, default to 0.

    Returns:
      res: a numpy array.
    """
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
    if pad_value is None:
        if len(shape) == 2:
            pad_value = image[0, 0]
        elif len(shape) == 3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    start = np.array(new_shape) / 2. - np.array(shape) / 2.
    if len(shape) == 2:
        res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1])] = image
    elif len(shape) == 3:
        res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
        int(start[2]):int(start[2]) + int(shape[2])] = image
    return res


def pad_and_or_crop(orig_data, new_shape, mode=None, coords=None):
    data = pad_if_too_small(orig_data, new_shape, pad_value=0)

    h, w = data.shape
    if mode == "centre":
        h_c = int(h / 2.)
        w_c = int(w / 2.)
    elif mode == "fixed":
        assert (coords is not None)
        h_c, w_c = coords
    elif mode == "random":
        h_c_min = int(new_shape[0] / 2.)
        w_c_min = int(new_shape[1] / 2.)

        if new_shape[0] % 2 == 1:
            h_c_max = h - 1 - int(new_shape[0] / 2.)
            w_c_max = w - 1 - int(new_shape[1] / 2.)
        else:
            h_c_max = h - int(new_shape[0] / 2.)
            w_c_max = w - int(new_shape[1] / 2.)

        h_c = np.random.randint(low=h_c_min, high=(h_c_max + 1))
        w_c = np.random.randint(low=w_c_min, high=(w_c_max + 1))

    h_start = h_c - int(new_shape[0] / 2.)
    w_start = w_c - int(new_shape[1] / 2.)
    data = data[h_start:(h_start + new_shape[0]), w_start:(w_start + new_shape[1])]

    return data, (h_c, w_c)



if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type=str, default="d:/data/acdc/acdc_contrastive/contrastive/2d/")
    parser.add_argument("--data_dir", type=str, default="/Contrast_Learining/ACDC_Contrast/positional_cl-main/acdc/ACDC_forReal_orig_Z/unlabeled/")
    parser.add_argument("--patch_size", type=tuple, default=(352, 352))
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--do_contrast", default=True, action='store_true')
    parser.add_argument("--slice_threshold", type=float, default=0.5)
    args = parser.parse_args()


    train_dataset = ACDC(keys=list(range(1,101)), purpose='train', args=args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=32,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    drop_last=False)

    pp = []
    for batch_idx, tup in enumerate(train_dataloader):
        print(f'the {batch_idx}th/{len(train_dataloader)} minibatch...')
        img1, img2, slice_position, partition = tup
        batch_size = img1.shape[0]
        # print(f'batch_size:{batch_size}, slice_position:{slice_position}')
        slice_position = slice_position.contiguous().view(-1, 1)
        mask = (torch.abs(slice_position.T.repeat(batch_size,1) - slice_position.repeat(1,batch_size)) < args.slice_threshold).float()
        # count how many positive pair in each batch
        for i in range(batch_size):
            pp.append(2*mask[i].sum()-1)
    pp = np.asarray(pp)
    pp_mean = np.mean(pp)
    pp_std = np.std(pp)
    print(f'average number of positive pairs mean:{pp_mean}, std:{pp_std}')




