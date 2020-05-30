import argparse
import os
import sys
import time

import cv2
import h5py
import imageio
import numpy as np
from skimage.morphology import dilation, erosion

from connectomics.data.utils import readh5, relabel, writeh5
from connectomics.utils.evaluation import adapted_rand


def get_args():
    parser = argparse.ArgumentParser(
        description='Specifications for segmentation.')
    parser.add_argument('-gt',  default='/home/gt_seg/',
                        help='path to groundtruth segmentation')
    parser.add_argument('-pd',  default='/home/pd_aff/',
                        help='path to predicted affinity graph')
    parser.add_argument('--aff-mode', type=int, default=0,
                        help='affinity post-processing method')
    parser.add_argument('--seg-mode', type=int, default=1,
                        help='segmentation method')
    parser.add_argument('--save', type=bool, default=False,
                        help='save segmentation')
    args = parser.parse_args()
    return args


args = get_args()

# add affinity location
D_aff = args.pd
aff = readh5(D_aff)

# scale affinity values and update type (if reqd)
print('dtype of affinity graph:', aff.dtype)
assert aff.dtype in [np.uint8, np.float32]
if aff.dtype == np.uint8:
    aff = (aff/255.0).astype(np.float32)

# post process affinity
if args.aff_mode == 1:
    print('applying x/y affinity erosion post-processing trick')
    temp = aff
    kernel = np.ones([3, 3], np.uint8)
    for j in range(aff.shape[0]):
        for k in range(aff.shape[1]):
            temp[j, k] = cv2.erode(
                aff[j, k], kernel, cv2.BORDER_CONSTANT, borderValue=0)
    aff = temp

# ground truth
# D0='/n/coxfs01/zudilin/research/mitoNet/data/file/snemi/label/'
D_seg = args.gt
suffix = D_seg.strip().split('.')[-1]
assert suffix in ['tif', 'h5']
if suffix == 'tif':
    seg = imageio.volread(D_seg).astype(np.uint32)
else:
    seg = readh5(D_seg).astype(np.uint32)

# print shape information
print('shape of affinity graph:', aff.shape)
print('shape of gt segmenation:', seg.shape)

if args.seg_mode == 0:
    # 3D zwatershed
    import zwatershed
    print('zwatershed:', zwatershed.__version__)

    st = time.time()

    # zwatershed parameters
    T_aff = [0.05, 0.995, 0.2]
    T_thres = [800]
    T_dust = 600
    T_merge = 0.9
    T_aff_rel = 1

    # segmentation
    out = zwatershed.zwatershed(
        aff, T_thres, T_aff=T_aff, T_dust=T_dust,
        T_merge=T_merge, T_aff_relative=T_aff_rel)[0][0]

    et = time.time()

    out = relabel(out)
    sn = '%s_%f_%f_%d_%f_%d_%f_%d.h5' % (
        args.seg_mode, T_aff[0], T_aff[1], T_thres[0],
        T_aff[2], T_dust, T_merge, T_aff_rel)

elif args.seg_mode == 1:
    # waterz
    import waterz
    print('waterz:', waterz.__version__)

    st = time.time()

    # waterz parameters
    low = 0.05
    high = 0.995
    mf = 'aff85_his256'
    T_thres = [0.6]

    # segmentation
    out = waterz.waterz(
        aff, T_thres, merge_function=mf, gt_border=0, fragments=None,
        aff_threshold=[low, high], return_seg=True, gt=seg)[0]

    et = time.time()

    out_copy = out
    out = relabel(out)
    sn = '%s_%f_%f_%f_%s.h5' % (args.seg_mode, low, high, T_thres[0], mf)

elif args.seg_mode == 2:
    # 2D zwatershed + waterz
    import waterz
    import zwatershed
    print('waterz:', waterz.__version__)
    print('zwatershed:', zwatershed.__version__)

    st = time.time()

    # zwatershed parameters
    T_thres = [150]
    T_aff = [0.05, 0.8, 0.2]
    T_dust = 150
    T_merge = 0.9
    T_aff_rel = 1

    # 2-D segmentation
    sz = np.array(aff.shape)
    out = np.zeros(sz[1:], dtype=np.uint64)
    id_st = np.uint64(0)
    # need to relabel the 2D seg, o/w out of bound
    for z in range(sz[1]):
        out[z] = relabel(zwatershed.zwatershed(aff[:, z:z+1], T_thres, T_aff=T_aff,
                                               T_dust=T_dust, T_merge=T_merge, T_aff_relative=T_aff_rel)[0][0])

        out[z][np.where(out[z] > 0)] += id_st
        id_st = out[z].max()

    # waterz parameters
    mf = 'aff50_his256'
    T_thres2 = [0.5]

    # 3D segmentation
    out = waterz.waterz(affs=aff, thresholds=T_thres2,
                        fragments=out, merge_function=mf)[0]

    et = time.time()

    sn = '%s_%f_%f_%d_%f_%d_%f_%d_%f_%s.h5' % (
        args.seg_mode, T_aff[0], T_aff[1], T_thres[0], T_aff[2], T_dust, T_merge, T_aff_rel, T_thres2[0], mf)

else:
    print('The segmentation method is not implemented yet!')
    raise NotImplementedError

# print time profile
print('time: %.1f s' % ((et-st)))

# ARAND evaluation
score = adapted_rand(out.astype(np.uint32), seg)
print('Adaptive rand: ', score)
# 0: 0.22
# 1: 0.098
# 2: 0.137

# save segmentation
if args.save:
    result_dir = os.path.dirname(args.pd) + '/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    writeh5(result_dir + sn, out, 'main')

python demo.py -p ../pytorch_connectomics/outputs/cerebellum_P0/test/seg2.h5 -gt ../pytorch_connectomics/outputs/cerebellum_P0/train/seg2_gt.h5 -ph ../pytorch_connectomics/outputs/cerebellum_P0/test/aff2.h5