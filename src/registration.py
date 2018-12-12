"""
The program is for registration the nii files to a fix size.
"""


from __future__ import print_function
from functools import reduce
from scipy.ndimage import zoom
from tqdm import tqdm
import numpy as np
import nibabel as nib
from os.path import join
import SimpleITK as sitk
import argparse
import sys
import os


def imread(img_path, t_sz):
    img = nib.load(img_path).get_data()
    sz = img.shape  # fixed size
    n_img = zoom(img, (t_sz[0]/sz[0],t_sz[1]/sz[1],t_sz[2]/sz[2]))
    try:
        assert n_img.shape == t_sz
    except:
        print(img_path, 'reszie error')
    n_img = sitk.GetImageFromArray(n_img)
    return sitk.Normalize(n_img)


def registation(fixed, img_move_path, img_out_path, target_sz, pixelType):
    moving = imread(img_move_path, target_sz)

    R = sitk.ImageRegistrationMethod()

    R.SetMetricAsJointHistogramMutualInformation()

    R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,
                                              numberOfIterations=100,
                                              convergenceMinimumValue=1e-5,
                                              convergenceWindowSize=5)

    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))

    R.SetInterpolator(sitk.sitkLinear)

    # R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

    outTx = R.Execute(fixed, moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    out = sitk.Cast(sitk.RescaleIntensity(out), pixelType)
    out = sitk.GetArrayFromImage(out)

    np.save(img_out_path, out)

def arg_parse():
    parser = argparse.ArgumentParser(description='generate the fixed image for registration')
    # parse command line options
    parser.add_argument('-corpus_dir', 
                        '--corpus_dir',
                        type=str, 
                        default='../Data/NII_data/',
                        help='The path to the nii directory')
    parser.add_argument('-fixed_name',
                        '--fixed_name',
                        type=str,
                        default='IDAGet_1.output-0384.nii',
                        help='The name of the fixed image')
    parser.add_argument('-output_prefix',
                        '--output_prefix',
                        type=str,
                        default='reg_',
                        help='The prefix of the ouptut files')
    parser.add_argument('-output_dir',
                        '--output_dir',
                        type=str,
                        default='../Data/Registration',
                        help='output directory')
    args = parser.parse_args()
    
    # fix the working directory
    CURR_DIR = os.getcwd()
    args.output_dir = join(CURR_DIR, args.output_dir)
    args.corpus_dir = join(CURR_DIR, args.corpus_dir)

    # create the output directory if it is not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args

def main(args):
    output_path = args.output_dir
    input_path = args.corpus_dir
    img_ref_path = join(input_path, args.fixed_name)
    target_sz = (128,128,128)
    pixelType = sitk.sitkInt16
    img_ref = imread(img_ref_path, target_sz)

    for i in tqdm(range(1, 1100)):
        img_move_path = join(input_path, 'IDAGet_1.output-'+ str(i).zfill(4) +'.nii')
        img_out_path = join(output_path, args.output_prefix + str(i).zfill(4))
        registation(img_ref, img_move_path, img_out_path, target_sz, pixelType)
        

if __name__ == "__main__":
    args = arg_parse()
    main(args)
    
