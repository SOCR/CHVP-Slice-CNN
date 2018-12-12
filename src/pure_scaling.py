"""
This program do the pure scaling preprocessing for all the nii files.
"""

import nibabel as nib
from scipy.ndimage import zoom
import scipy.io
from os.path import join
import os
import numpy as np
from tqdm import tqdm
import argparse


def pure_scaling(img_path, t_sz):
	img = nib.load(img_path).get_data()
	sz = img.shape  # fixed size
	n_img = zoom(img, (t_sz[0]/sz[0],t_sz[1]/sz[1],t_sz[2]/sz[2]))
	try:
		assert n_img.shape == t_sz
	except:
		print(img_path, 'reszie error')

	return n_img

def main(args):
	target_sz = (128,128,128)
	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)

	for i in tqdm(range(1, 1100)):
		filename = 'IDAGet_1.output-'+ str(i).zfill(4) +'.nii'
		output_name = args.output_prefix + str(i).zfill(4)
		img = pure_scaling(join(args.corpus_dir, filename), target_sz)
		np.save(join(args.output_dir, output_name), img)

def arg_parse():
    parser = argparse.ArgumentParser(description='generate the fixed image for registration')
    # parse command line options
    parser.add_argument('-corpus_dir', 
                        '--corpus_dir',
                        type=str, 
                        default='../Data/NII_data/',
                        help='The path to the nii directory')
    parser.add_argument('-output_prefix',
                        '--output_prefix',
                        type=str,
                        default='pure_',
                        help='The prefix of the ouptut files')
    parser.add_argument('-output_dir',
                        '--output_dir',
                        type=str,
                        default='../Data/Pure_Scaling',
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

if __name__ == '__main__':
	args = arg_parse()
	main(args)
