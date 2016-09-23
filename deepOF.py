import os, sys
import argparse
# from ucf101train import train
from train import train

def deepOF(data_path):
	MPISintel = True
	UCF101 = False

	if MPISintel:
		image_size = [256, 512]
		crop_size = [224, 480]
		split = 0.8
		passKey = 'final'		# clean or final
		train(data_path, image_size, split, passKey)
	elif UCF101:
		image_size = [256, 320]
		train(data_path, image_size)

def main():
	# Example usage: python deepOF.py ~/Documents/MPI-Sintel/
	parser = argparse.ArgumentParser(description='Unsupervised motion estimation from videos')
	parser.add_argument('data_path', type=str, help='Path to your data')
	args = parser.parse_args()

	deepOF(args.data_path)
    
if __name__ == '__main__':
    main()











