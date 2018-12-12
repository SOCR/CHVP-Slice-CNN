"""
The program is going to extract the labels from the metadata (an excel file)
and relabel all the preprocessing npy data in the first step
"""

from os.path import join
import pandas as pd
import numpy as np
import os
import argparse


def load_label(excel_path):
	dic = {}
	xl = pd.ExcelFile(excel_path)
	df = xl.parse('ABIDE_Aggregated_Data')
	for row in df.iterrows():
		img_idx = row[1]['Data'].split('/')[-1][-8:-4]
		raw_label = row[1]['dx']
		label = '0' if raw_label == 1 else '1'
		dic[img_idx] = label
	
	return dic

def relabel(img_dir, labels):
	for filename in os.listdir(img_dir):
		if filename.endswith('.npy') and not (filename.endswith('_1.npy') or filename.endswith('_0.npy')):
			img_idx = filename.split('.')[0].split('_')[-1]
			prefix = filename.split('_')[0]
			try:
				label = labels[img_idx]
			except:
				print('label not found:', img_idx)
				os.remove(join(img_dir, filename))
				continue

			new_filename = '_'.join((prefix, img_idx, label)) + '.npy'
			os.rename(join(img_dir, filename), join(img_dir, new_filename))
			
def main(args):
	excel_dir = args.excel_dir
	excel_name = args.excel_name
	img_dir = args.img_dir

	labels = load_label(join(excel_dir, excel_name))
	relabel(img_dir, labels)

def arg_parse():
    parser = argparse.ArgumentParser(description='generate the fixed image for registration')
    # parse command line options
    parser.add_argument('-excel_dir', 
                        '--excel_dir',
                        type=str, 
                        default='../Data/',
                        help='The path to the excel file directory')
    parser.add_argument('-excel_name',
                        '--excel_name',
                        type=str,
                        default='ABIDE_AggregaredData_Dictionary.xlsx',
                        help='The name of the excel')
    parser.add_argument('-img_dir',
                        '--img_dir',
                        type=str,
                        default='../Data/Pure_Scaling',
                        help='The prefix of the ouptut files')
    args = parser.parse_args()
    
    # fix the working directory
    CURR_DIR = os.getcwd()
    args.excel_dir = join(CURR_DIR, args.excel_dir)
    args.img_dir = join(CURR_DIR, args.img_dir)

    return args

if __name__ == "__main__":
	args = arg_parse()
	main(args)
