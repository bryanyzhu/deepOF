import os, sys
import argparse
from trainOF import train  

def main():
	"""
	Example usage: 
	python deepOF.py flyingChairs ~/Documents/FlyingChairs_release/ 4 0.000016 0.5 18 18 /tmp/integrate_v1/
	"""
	parser = argparse.ArgumentParser(description='Unsupervised motion estimation from videos.')
	parser.add_argument('dataset', type=str, help='Dataset name.')
	parser.add_argument('data_path', type=str, help='Path to your data.')
	parser.add_argument('batch_size', type=int, help='The number of images in each batch.')
	parser.add_argument('learning_rate', type=float, help='The initial learning rate.')
	parser.add_argument('lr_decay', type=float, help='Learning rate decay factor.')
	parser.add_argument('num_epochs_per_decay', type=int, help='Number of epochs after which learning rate decays.')
	parser.add_argument('save_interval_epoch', type=int, help='The frequency with which the model is saved, in epoch.')
	parser.add_argument('log_dir', type=str, help='Directory where to write event logs.')
	args = parser.parse_args()

	opts = {}
	opts["dataset"] = args.dataset
	opts["data_path"] = args.data_path
	opts["batch_size"] = args.batch_size
	opts["learning_rate"] = args.learning_rate
	opts["lr_decay"] = args.lr_decay
	opts["num_epochs_per_decay"] = args.num_epochs_per_decay
	opts["save_interval_epoch"] = args.save_interval_epoch
	opts["log_dir"] = args.log_dir

	train(opts)
    
if __name__ == '__main__':
    main()










