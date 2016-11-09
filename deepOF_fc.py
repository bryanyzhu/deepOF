import os, sys
import argparse
from flyingChairsTrain_vgg import train

def deepOF(data_path):
	image_size = [320, 448]
	train(data_path, image_size)

def main():
	# Example usage: python deepOF_fc.py /home/yzhu25/Documents/FlyingChairs_release/
	parser = argparse.ArgumentParser(description='Unsupervised motion estimation from videos')
	parser.add_argument('data_path', type=str, help='Path to your data')
	args = parser.parse_args()

	deepOF(args.data_path)
    
if __name__ == '__main__':
    main()










