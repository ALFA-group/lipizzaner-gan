import logging
import os
import pathlib
import sys
from datetime import datetime


class LogHelper:
    @staticmethod
    def setup(log_level, output_dir):
        if not os.path.exists(output_dir):
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger()
        logger.setLevel(logging.getLevelName(log_level))
        file_handler = logging.FileHandler(
            datetime.now().strftime("{}/lipizzaner_%Y-%m-%d_%H-%M.log".format(output_dir))
        )
        console_handler = logging.StreamHandler()

        # Formatter
        formatter = logging.Formatter("%(asctime)s %(levelname)s - %(name)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Clear handlers from possible previous experiments to avoid duplicate logging
        logger.handlers = []
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        sys.excepthook = lambda *ex: logger.critical("Unhandled exception", exc_info=ex)

    @staticmethod
    def log_only_flask_warnings():
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.WARNING)
