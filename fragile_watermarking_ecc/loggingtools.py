from __future__ import print_function
import os
HOME = os.environ["HOME"]

import sys
import logging
from logging.handlers import TimedRotatingFileHandler


# https://gist.github.com/nguyenkims/e92df0f8bd49973f0c94bddf36ed7fd0
FORMATTER = logging.Formatter("%(asctime)s - [%(process)d] - %(levelname)s - %(message)s - pysage %(name)s")
LOG_FILE = os.path.join(HOME, "tmp/fwecc.log")



def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler

def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.setFormatter(FORMATTER)
    return file_handler

def get_logger(logger_name):
    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.DEBUG) # better to have too much log than not enough

    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())

    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger

# https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings
def setup_logger(name, log_file, level=logging.INFO,
                 use_console_handler=False):
    """Function setup as many loggers as you want"""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = TimedRotatingFileHandler(log_file, when='midnight')
    handler.setFormatter(FORMATTER)

    logger.addHandler(handler)
    if use_console_handler:
        console_handler = get_console_handler()
        logger.addHandler(console_handler)

    return logger


#
# class MyLogger:
#     def __init__(self, logger_name, log_fname, formatter=FORMATTER):
#         self.logger_name = logger_name
#         self.log_fname = log_fname
#         self.formatter = formatter
#         self.logger = self.get_logger()
#
#     def get_console_handler(self):
#         console_handler = logging.StreamHandler(sys.stdout)
#         console_handler.setFormatter(self.formatter)
#         return console_handler
#
#     def get_file_handler(self):
#         file_handler = TimedRotatingFileHandler(self.log_fname,
#                                                 when='midnight')
#         file_handler.setFormatter(self.formatter)
#         return file_handler
#
#     def get_logger(self):
#         logger = logging.getLogger(self.logger_name)
#         # better to have too much log than not enough
#         logger.setLevel(logging.DEBUG)
#
#         logger.addHandler(get_console_handler())
#         logger.addHandler(get_file_handler())
#
#         # with this pattern, it's rarely necessary to
#         # propagate the error up to parent
#         logger.propagate = False
#         return logger
#
#     def info(self, msg):
#         self.logger.info(msg)
#
#     def debug(self, msg):
#         self.logger.debug(msg)
#
#     def warning(self, msg):
#         self.logger.warning(msg)
#

def main():
    pass


if __name__ == "__main__":
    sys.exit(main())