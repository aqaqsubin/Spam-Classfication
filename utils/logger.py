from .data_util import mkdir_p

import logging 
import logging.handlers

from os.path import join as pjoin

class Logger:
    def __init__(self, logger_name, dirpath=None):
        mkdir_p(dirpath)

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        file_handler = logging.handlers.TimedRotatingFileHandler( 
        filename=pjoin(dirpath, 'api.log'), when='midnight', interval=1, backupCount=31, encoding='utf-8') 
        
        file_handler.suffix = '%Y%m%d.log' 
        formatter = logging.Formatter('%(asctime)s | %(levelname)s >> %(message)s') 
        
        file_handler.setFormatter(formatter) 

        if len(self.logger.handlers) == 0:
            self.logger.addHandler(file_handler) 
 
    def info(self, query):
        self.logger.info('{}'.format(query))

    def warning(self, query):
        self.logger.warning('{}'.format(query))

    def error(self, query):
        self.logger.error('{}'.format(query))

    def debug(self, query):
        self.logger.debug('{}'.format(query))    
