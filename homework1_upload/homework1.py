from MyTool.HackTool import HackTool
from MyTool.DataManager import DataManager
def function(givingValue):
	print(givingValue)
	pass
if __name__ == '__main__':
    #initialize all the tools
    myDataManager = DataManager('timing_noisy.csv')
    myHackTool = HackTool()

    #start the processing
    myHackTool._maxCoreProcessing(function,[givingValue for givingValue in range(0, 10000)])
