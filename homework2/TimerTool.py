
import time
import platform
import psutil
from multiprocessing import Pool


class TimerTool:
    def __init__(self):
        self.__startTime = 0
        self.__executionTime = 0
        self.__number_of_tasks = 0
        self._start()
        self._cores_count = psutil.cpu_count()
        print("----------🛠️ Your System Information------")
        print("🖥️  	System: ", platform.system(), "		|")
        print("🖲️  	processor: ", platform.processor(), "		|")
        print("🎛️  	CPU cores: ", psutil.cpu_count(), "			|")
        print("💾  	RAM: ", psutil.virtual_memory().total / (1024.0 ** 3), "GB", "			|")
        print("-----------------------------------------\n")
        self._stop()

    def _maxCoreProcessing(self, function, values):
        self.__number_of_tasks = len(values)
        print(f"----------📥 MaxCoreProcessing Started-----------")
        print(f"⛏️ 	Function Name:{function.__name__}		|")
        print(f"📦 	Number of Tasks:{str(self.__number_of_tasks)}			|")
        self._start()
        with Pool() as p:
            p.map(function, values)
        pass
        self._stop()
        print(f"----------✅ MaxCoreProcessing Ended-------------\n")
        self._display()

    def _start(self):
        self.__startTime = time.time()

    def _stop(self):
        self.__executionTime = time.time() - self.__startTime

    def _display(self):
        print("----------🛠️  Execution Information-------------------------------")
        print("🚀 Execution Time ("+str(psutil.cpu_count())+" cores used):	" + str(self.__executionTime)+"'s	|")
        print("🎛️  Each Core's Execution Time:		" + str(self.__executionTime/psutil.cpu_count())+"'s	|")
        print("-----------------------------------------------------------------\n")


if __name__ == '__main__':
    myHackTool = TimerTool()
    myHackTool._display()
