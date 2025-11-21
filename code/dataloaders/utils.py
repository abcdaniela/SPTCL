import os

import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
from skimage import measure


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


def get_cityscapes_labels():
    return np.array([
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [
                           0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [
                           64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    # for key, val in param.items():
    #     log_file.write(key + ':' + str(val) + '\n')
    log_file.write(str(param))
    log_file.close()


def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(
            weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss


def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def get_iou(pred, gt, n_classes=21):
    total_iou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou

    return total_iou


def get_dice(pred, gt):
    total_dice = 0.0
    pred = pred.long()
    gt = gt.long()
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]
        dice = 2.0*torch.sum(pred_tmp*gt_tmp).item()/(1.0 +
                                                      torch.sum(pred_tmp**2)+torch.sum(gt_tmp**2)).item()
        print(dice)
        total_dice += dice

    return total_dice


def get_mc_dice(pred, gt, num=2):
    # num is the total number of classes, include the background
    total_dice = np.zeros(num-1)
    pred = pred.long()
    gt = gt.long()
    for i in range(len(pred)):
        for j in range(1, num):
            pred_tmp = (pred[i] == j)
            gt_tmp = (gt[i] == j)
            dice = 2.0*torch.sum(pred_tmp*gt_tmp).item()/(1.0 +
                                                          torch.sum(pred_tmp**2)+torch.sum(gt_tmp**2)).item()
            total_dice[j-1] += dice
    return total_dice


def post_processing(prediction):
    prediction = nd.binary_fill_holes(prediction)
    label_cc, num_cc = measure.label(prediction, return_num=True)
    total_cc = np.sum(prediction)
    measure.regionprops(label_cc)
    for cc in range(1, num_cc+1):
        single_cc = (label_cc == cc)
        single_vol = np.sum(single_cc)
        if single_vol/total_cc < 0.2:
            prediction[single_cc] = 0

    return prediction


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
