from abc import ABC, abstractmethod


class Logger(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def initialize(self, p): 
        pass


    # set some parameters in the logger, for instance to load the eval metrics
    # of some checkpointed run
    @abstractmethod
    def set_values(self, value_dict): 
        pass
    
    # will add data to 'eval' subdictionnary
    @abstractmethod
    def log(self, data_type, data):
        pass
    
    # flush and stop
    @abstractmethod
    def stop(self):
        pass
