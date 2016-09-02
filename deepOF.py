import os, sys
import argparse
from train import train

def deepOF(data_path):
	image_size = [128, 160]
	split = 0.8
	passKey = 'final'		# clean or final
	train(data_path, image_size, split, passKey)
	# sintel = sintelLoader(data_path, image_size, split, passKey)

def main():
	# Example usage: python deepOF.py ~/Documents/MPI-Sintel/
	parser = argparse.ArgumentParser(description='Unsupervised motion estimation from videos')
	parser.add_argument('data_path', type=str, help='Path to your data')
	args = parser.parse_args()
	print args

	deepOF(args.data_path)
    
if __name__ == '__main__':
    main()











