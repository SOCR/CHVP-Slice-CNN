from os.path import join
import os
import random
import shutil
import argparse

# DATA_SOURCE = '../../../Data/Pure_Scaling'
# TARGET_DIR = '../corpus'

def train_test_split(ratio, prefix, data_source, corpus_path):
	set0, set1 = determine_pool(data_source)
	train_set0 = set(random.sample(set0, int(ratio * len(set0))))
	train_set1 = set(random.sample(set1, int(ratio * len(set1))))

	train_dir_name = '_'.join(('train_data', prefix))
	test_dir_name = '_'.join(('test_data', prefix))

	clear_corpus_path(corpus_path, train_dir_name, test_dir_name)

	move_data(train_set0, '0', prefix, data_source, join(corpus_path, train_dir_name))
	move_data(train_set1, '1', prefix, data_source, join(corpus_path, train_dir_name))
	move_data(set0-train_set0, '0', prefix, data_source, join(corpus_path, test_dir_name))
	move_data(set1-train_set1, '1', prefix, data_source, join(corpus_path, test_dir_name))

def move_data(file_set, label, prefix, data_source, corpus_path):
	for file_idx in file_set:
		filename = '_'.join((prefix, file_idx, label)) + '.npy'
		shutil.copyfile(join(data_source, filename), join(corpus_path, filename))


def clear_corpus_path(corpus_path, train_dir_name, test_dir_name):
	try:
		shutil.rmtree(join(corpus_path, train_dir_name))
		shutil.rmtree(join(corpus_path, test_dir_name))
	except:
		pass
		
	os.makedirs(join(corpus_path, train_dir_name))
	os.makedirs(join(corpus_path, test_dir_name))

def determine_pool(data_source):
	set0, set1 = set(), set()
	for filename in os.listdir(data_source):
		if filename.endswith('.npy'):
			img_idx = filename.split('.')[0].split('_')[1]
			label = filename.split('.')[0].split('_')[2]
			if label == '0':
				set0.add(img_idx)
			else:
				set1.add(img_idx)

	return set0, set1

def arg_parse():
	parser = argparse.ArgumentParser(description='train test split parameters')
	# parse command line options
	parser.add_argument('-ratio', '--ratio', type=float, help='ratio of the training set')
	parser.add_argument('-prefix', '--prefix', type=str, help='The prefix of the data (pure or reg)')
	parser.add_argument('-data_source', '--data_source', type=str, help='path to the relabeled np files')
	parser.add_argument('-corpus_path', '--corpus_path', type=str, help='output path')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = arg_parse()
	train_test_split(ratio=args.ratio, prefix=args.prefix, data_source=args.data_source, corpus_path=args.corpus_path)