
import logging
import os
from datetime import datetime
from configs.path_config import PathConfig


class Logger:
    def __init__(self, name="PLAA_MLLM", log_dir=None):
        self.logger = None
        self.log_dir = log_dir
        
    def info(self, msg):
        pass
    
    def debug(self, msg):
        pass
    
    def warning(self, msg):
        pass
    
    def error(self, msg):
        pass
    
    def critical(self, msg):
        pass
    
    def _setup_logger(self):
        pass
