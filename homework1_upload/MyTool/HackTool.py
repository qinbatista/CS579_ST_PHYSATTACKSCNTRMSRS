
import time
import platform
import psutil
from multiprocessing import Pool


class HackTool:
    def __init__(self):
        self.__startTime = 0
        self.__executionTime = 0
        self.__number_of_tasks = 0
        self._timerStart()
        print("----------🛠️ Your System Information------")
        print("🖥️  	System: ", platform.system(), "		|")
        print("🖲️  	rocessor: ", platform.processor(), "		|")
        print("🎛️  	CPU cores: ", psutil.cpu_count(), "			|")
        print("💾  	RAM: ", psutil.virtual_memory().total / (1024.0 ** 3), "GB", "			|")
        print("-----------------------------------------\n")
        self._timerStop()

    def _maxCoreProcessing(self, function, values):
        self.__number_of_tasks = len(values)
        print(f"----------📥 MaxCoreProcessing Started-----------")
        print(f"⛏️ 	Function Name:{function.__name__}		|")
        print(f"📦 	Number of Tasks:{str(self.__number_of_tasks)}			|")
        self._timerStart()
        with Pool() as p:
            p.map(function, values)
        pass
        self._timerStop()
        print(f"----------✅ MaxCoreProcessing Ended-------------\n")
        self._displayExecutionTime()

    def _timerStart(self):
        self.__startTime = time.time()

    def _timerStop(self):
        self.__executionTime = time.time() - self.__startTime

    def _displayExecutionTime(self):
        print("----------🛠️  Execution Information-------------------------------")
        print("🚀 Execution Time ("+str(psutil.cpu_count())+" cores used):	" + str(self.__executionTime)+"'s	|")
        print("🎛️  Each Core's Execution Time:		" + str(self.__executionTime/psutil.cpu_count())+"'s	|")
        print("-----------------------------------------------------------------\n")


if __name__ == '__main__':
    myHackTool = HackTool()
    myHackTool._displayExecutionTime()
