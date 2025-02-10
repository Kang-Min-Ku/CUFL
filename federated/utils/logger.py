import logging
import os
import sys
import inspect
import re
import warnings

class Logger:
    def __init__(self, mode="default"):
        """
        Class Logger is the class which is easy to use logging module.

        <parameter explanation>
        *mode: str
            mode defines the way to use logging module. I provide to modes "default" and "direct".
                default: This mode is using logging module.
                direct: This mode is using write function.
        """
        assert mode in ["default", "direct"], "specify appropriate mode"
        self.mode = mode
        self.logger = None
        self.fd_dict = {}

        self.frame = inspect.stack()[1]
        self.caller_file_path = self.frame.filename
        self.home_path = os.path.expanduser("~")

    def set_basic_config(self, **config):
        """
        Function set_basic_config is the function which is setting basic config of logging module.
        To use set_basic_config you should specify argument with its name e.g. filename="log.log".

        <parameter(config) explanation>
        *filename: str
            filename is the name of log file including directory path.
        *filemode: str
            filemode is the mode of log file. "w" or "a"
        *format: str
            format is the format of log message.
        *datefmt: str
            datefmt is the format of date in log message.
        *level: int
            level is the level of log message.

        """
        # if not __name__ != "__main__":
        #     return
        filename = config.get("filename", "log.log")
        filemode = config.get("filemode", "a")
        format = config.get("format", "%(asctime)s %(levelname)s:%(message)s")
        datefmt = config.get("datefmt", "%Y/%m/%d %H:%M")
        level = config.get("level", logging.INFO)

        logging.basicConfig(
            filename=filename,
            filemode=filemode,
            format=format,
            datefmt=datefmt,
            level=level
        )

    def set_logger(self, **config):
        """
        Function set_logger is the function which is setting logger of logging module.
        To use set_logger you should specify argument with its name e.g. filename="log.log".

        <parameter(config) explanation>
        *loggername: str
            loggername is the name of logger.
            If loggername is not specified, name of logger is set to relative path of called file splitted by '.'
            e.g. toolbox/record/logging.py -> toolbox.record.logging
        *level: int
            level is the level of log message.
        """
        loggername = config.get("loggername", self.get_default_logger_name())
        level = config.get("level", logging.INFO)

        existing_loggers = [logging.getLogger(name).name for name in logging.root.manager.loggerDict]
        if loggername in existing_loggers:
            warning = f'logger "{loggername}" already exists. Same named logger can cause unexpected behavior.\n'
            warning += f'\t\t\t\t\t\te.g. log overlapping for specific file\n'
            warning += f'\t\t\t\t\t\tDifferent loggername is recommended.'
            warnings.warn(warning, UserWarning)

        self.logger = logging.getLogger(loggername)
        self.logger.setLevel(level)
        
    def set_handler(self, mode="file", **config):
        """
        Function set_handler is the function which is setting handler of logging module.
        To use set_handler you should specify argument with its name e.g. filename="log.log".

        <parameter(config) explanation>
        *mode: str
            mode defines the way to log. I provide to modes "file" and "stream".
        *filename: str
            filename is the name of log file including directory path.
        *filemode: str
            filemode is the mode of log file. "w" or "a"
        *format: str
            format is the format of log message.
        *datefmt: str
            datefmt is the format of date in log message.
        *level: int
            level is the level of log message.
        """
        assert mode in ["file", "stream"], "specify appropriate mode"
        assert self.logger is not None, "set logger first"

        level = config.get("level", logging.INFO)
        filename = config.get("filename", f"log_{level}.log")
        filemode = config.get("filemode", "a")
        format = config.get("format",
                            "%(asctime)s:%(levelname)s:%(name)s:%(message)s")
        datefmt = config.get("datefmt", "%Y/%m/%d %H:%M")

        if mode == "file":
            handler = logging.FileHandler(filename, mode=filemode)
        elif mode == "stream":
            handler = logging.StreamHandler()

        formatter = logging.Formatter(format, datefmt=datefmt)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if self.mode == "direct":
            fd = open(filename, "w")
            self.fd_dict[filename] = fd

    def wrapup_handler(self):
        for fd in self.fd_dict.values():
            fd.close()

    def get_default_logger_name(self):
        logger_path = re.sub(self.home_path, "", self.caller_file_path)
        logger_path = logger_path.strip("/")
        logger_name = logger_path.replace("/", ".")
        logger_name = logger_name[:logger_name.rfind(".")]

        return logger_name

    def info(self, msg):
        if self.mode == "default":
            self.logger.info(msg)
        elif self.mode == "direct":
            self[logging.INFO].write(msg + "\n")

    def debug(self, msg):
        if self.mode == "default":
            self.logger.debug(msg)
        elif self.mode == "direct":
            self[logging.DEBUG].write(msg + "\n")

    def warning(self, msg):
        if self.mode == "default":
            self.logger.warning(msg)
        elif self.mode == "direct":
            self[logging.WARNING].write(msg + "\n")

    def error(self, msg):
        if self.mode == "default":
            self.logger.error(msg)
        elif self.mode == "direct":
            self[logging.ERROR].write(msg + "\n")

    def critical(self, msg):
        if self.mode == "default":
            self.logger.critical(msg)
        elif self.mode == "direct":
            self[logging.CRITICAL].write(msg + "\n")