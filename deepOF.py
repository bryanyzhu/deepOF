import os, sys
import argparse
# from flyingChairsTrain import train
# from ucf101train import train
from sintelTrain import train

def deepOF(data_path):
	MPISintel = True
	ucf101 = False
	flyingChairs = False

	if MPISintel:
		time_step = 5
		image_size = [256, 512]
		crop_size = [224, 480]
		split = 0.8
		passKey = 'final'		# clean or final
		train(data_path, image_size, split, time_step, passKey)
	elif ucf101:
		image_size = [320, 384]
		train(data_path, image_size)
	elif flyingChairs:
		image_size = [320, 448]
		train(data_path, image_size)


def main():
	# Example usage: python deepOF.py ~/Documents/MPI-Sintel/
	parser = argparse.ArgumentParser(description='Unsupervised motion estimation from videos')
	parser.add_argument('data_path', type=str, help='Path to your data')
	args = parser.parse_args()

	deepOF(args.data_path)
    
if __name__ == '__main__':
    main()










