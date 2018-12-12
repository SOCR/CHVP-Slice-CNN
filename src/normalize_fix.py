import nibabel as nib
from scipy.ndimage import zoom
import scipy.io
from os.path import join
import numpy as np
import SimpleITK as sitk
import argparse

def main(args):
	fixed_name 	= args.fixed_name
	corpus = args.corpus_dir
	target_dir = args.output_dir
	target_name = args.output_name
	t_sz = (128, 128, 128)		# target size

	fixed = nib.load(join(corpus, fixed_name)).get_data()

	f_sz = fixed.shape	# fixed size
	n_fixed = zoom(fixed, (t_sz[0]/f_sz[0],t_sz[1]/f_sz[1],t_sz[2]/f_sz[2]))


	# np.save(target_name, normalized_fixed)

	# scipy.io.savemat(join(target_dir, target_name +".mat"), mdict={"arr": normalized_fixed})
	n_fixed = sitk.GetImageFromArray(n_fixed)
	n_fixed = sitk.Cast(sitk.RescaleIntensity(n_fixed), sitk.sitkInt16)
	sitk.WriteImage(n_fixed, join(target_dir, target_name +".nii"))

def arg_parse():
	parser = argparse.ArgumentParser(description='generate the fixed image for registration')
	# parse command line options
	parser.add_argument('-corpus_dir', 
						'--corpus_dir',
						type=str, 
						default='../Data/NII_Data',
						help='The path to the nii directory')
	parser.add_argument('-fixed_name',
						'--fixed_name',
						type=str,
						default='IDAGet_1.output-0384.nii',
						help='The name of the fixed image')
	parser.add_argument('-output_name',
						'--output_name',
						type=str,
						default='normalized_fixed',
						help='The name of the output image')
	parser.add_argument('-output_dir',
						'--output_dir',
						type=str,
						default='.',
						help='output directory')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = arg_parse()