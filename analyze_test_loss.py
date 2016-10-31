import os,sys
import numpy as np
import matplotlib.pyplot as pp

log_file = "/home/yzhu25/flownet_smooth_trial2.txt"
# log_file = "/home/yzhu25/Documents/deepOF/no_scale_no_smooth.txt"
f_log = open(log_file, "r")
log_lines = f_log.readlines()
print len(log_lines)

test_loss_all = []
for line_num in xrange(len(log_lines)):
	line = log_lines[line_num]
	if "***Test: E" in line:
		line_info = line.split()
		# print line_info
		test_loss = line_info[6]
		test_loss_all.append(test_loss)
		# print test_loss

losses = np.asarray(test_loss_all)
pp.plot(losses, 'x')
pp.show()
		

