
import pandas as pd
class DataManager:
    def __init__(self,path):
        self._csv_data = pd.read_csv(path)


