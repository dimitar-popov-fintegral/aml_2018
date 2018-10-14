import os
from enum import Enum 

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

#######################################################################
class Tasks(Enum):

	TASK0 = 'task0'
	TASK1 = 'task1'
	TASK2 = 'task2'


#######################################################################
def data_dir():

	return os.path.join(THIS_DIR, 'store')


#######################################################################
def output_dir():

	if not os.path.isdir(os.path.join(THIS_DIR, 'output')):
		os.mkdir(os.path.join(THIS_DIR, 'output'), mode=770)

	return os.path.join(THIS_DIR, 'output')
	