import sys
from string import join
filename=raw_input('Enter the file name:')
fname=filename+".netlist"
try :
	with open(fname) as f:     #Open file
		content = f.readlines()    # Read the lines 
	content = [x.strip() for x in content]  # Remove whitespaces
except:
	print('Input valid file name')



