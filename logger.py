import logging
import sys
import os
from config import LOG_LEVEL, LOG_FORMAT, BASE_DIR

def get_logger(name: str):
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if the logger is already configured
    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)
        formatter = logging.Formatter(LOG_FORMAT)
        
        # 1. Console Handler (for real-time terminal viewing)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 2. File Handler (for persistence and debugging)
        # Writes to assignment/app.log
        log_file = os.path.join(BASE_DIR, "app.log")
        file_handler = logging.FileHandler(log_file, mode='a') # 'a' for append
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

# Default app-wide logger
logger = get_logger("apple-expert-squad")
