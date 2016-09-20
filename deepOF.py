import os, sys
import argparse
from train import train
# from ucf101Loader import ucf101Loader

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
		image_size = [240, 320]
		dl = ucf101Loader(data_path, image_size)
		a, b, c = dl.sampleTrain(4)



def main():
	# Example usage: python deepOF.py ~/Documents/MPI-Sintel/
	parser = argparse.ArgumentParser(description='Unsupervised motion estimation from videos')
	parser.add_argument('data_path', type=str, help='Path to your data')
	args = parser.parse_args()

	deepOF(args.data_path)
    
if __name__ == '__main__':
    main()











