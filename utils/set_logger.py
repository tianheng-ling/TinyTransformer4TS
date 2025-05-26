import logging


def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if not logger.hasHandlers():
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(file_formatter)
        logger.addHandler(stream_handler)

    return logger
