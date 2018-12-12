import nibabel as nib
import scipy.io
from os.path import join
import os
import numpy as np
from tqdm import tqdm

DATA_SOURCE = '../Data/NII_data'

def pure_scaling(img_path, t_sz):
	img = nib.load(img_path).get_data()
	sz = img.shape  # fixed size
	n_img = zoom(img, (t_sz[0]/sz[0],t_sz[1]/sz[1],t_sz[2]/sz[2]))
	try:
		assert n_img.shape == t_sz
	except:
		print(img_path, 'reszie error')

	return n_img

if __name__ == '__main__':
	shape_set = set()
	for i in tqdm(range(1, 1100)):
		filename = 'IDAGet_1.output-'+ str(i).zfill(4) +'.nii'
		img = nib.load(join(DATA_SOURCE, filename)).get_data()
		shape_set.add(img.shape)

	print(len(shape_set))
	print(shape_set)
