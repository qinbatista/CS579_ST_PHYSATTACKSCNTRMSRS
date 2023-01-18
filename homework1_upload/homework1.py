from MyTool.HackTool import HackTool
from MyTool.DataManager import DataManager
from multiprocessing import Pool
import numpy as np
def myFunction(givingValue):
	np.show_config()
	#generate 1 million np random numbers
	pass
if __name__ == '__main__':
    #initialize all the tools
    myDataManager = DataManager('timing_noisy.csv')
    myHackTool = HackTool()

    #start the processing
    myHackTool._maxCoreProcessing(myFunction,[givingValue for givingValue in range(0, 1)])
