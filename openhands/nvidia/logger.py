import logging
import os
import re
from typing import Mapping

from openhands.core.logger import (
    LOG_COLORS,
    ColoredFormatter,
    ColorType,
    LlmFileHandler,
    NoColorFormatter,
    OpenHandsLoggerAdapter,
    SensitiveDataFilter,
    StackInfoFilter,
    json_formatter,
    json_log_handler,
)

# NVIDIA-specific environment variables
NV_LOG_LEVEL = os.getenv('NV_LOG_LEVEL', 'INFO').upper()
NV_DEBUG = os.getenv('NV_DEBUG', 'False').lower() in ['true', '1', 'yes']
NV_LOG_JSON = os.getenv('NV_LOG_JSON', 'False').lower() in ['true', '1', 'yes']
NV_LOG_TO_FILE = os.getenv('NV_LOG_TO_FILE', 'False').lower() in ['true', '1', 'yes']
NV_LOG_ALL_EVENTS = os.getenv('NV_LOG_ALL_EVENTS', 'False').lower() in [
    'true',
    '1',
    'yes',
]

if NV_DEBUG:
    NV_LOG_LEVEL = 'DEBUG'

# NVIDIA-specific log colors (extending the base colors)
NV_LOG_COLORS: Mapping[str, ColorType] = {
    **LOG_COLORS,  # Inherit all base colors
    'GPU': 'light_blue',
    'CUDA': 'light_cyan',
}


class NvColoredFormatter(ColoredFormatter):
    """NVIDIA-specific colored formatter that uses NV color mappings."""

    def format(self, record: logging.LogRecord) -> str:
        msg_type = record.__dict__.get('msg_type', '')
        event_source = record.__dict__.get('event_source', '')
        if event_source:
            new_msg_type = f'{event_source.upper()}_{msg_type}'
            if new_msg_type in NV_LOG_COLORS:
                msg_type = new_msg_type
        if msg_type in NV_LOG_COLORS:
            # Temporarily override the global LOG_COLORS for this format call
            import openhands.core.logger

            original_colors = openhands.core.logger.LOG_COLORS
            original_disable_color = openhands.core.logger.DISABLE_COLOR_PRINTING
            original_log_all_events = openhands.core.logger.LOG_ALL_EVENTS
            original_debug = openhands.core.logger.DEBUG

            openhands.core.logger.LOG_COLORS = NV_LOG_COLORS
            openhands.core.logger.DISABLE_COLOR_PRINTING = False
            openhands.core.logger.LOG_ALL_EVENTS = NV_LOG_ALL_EVENTS
            openhands.core.logger.DEBUG = NV_DEBUG

            result = super().format(record)

            # Restore original values
            openhands.core.logger.LOG_COLORS = original_colors
            openhands.core.logger.DISABLE_COLOR_PRINTING = original_disable_color
            openhands.core.logger.LOG_ALL_EVENTS = original_log_all_events
            openhands.core.logger.DEBUG = original_debug

            return result
        return super().format(record)


class NvSensitiveDataFilter(SensitiveDataFilter):
    """NVIDIA-specific sensitive data filter with additional patterns."""

    def filter(self, record: logging.LogRecord) -> bool:
        # First apply the base filter
        result = super().filter(record)

        # Add NVIDIA-specific sensitive patterns
        msg = record.getMessage() if hasattr(record, 'getMessage') else str(record.msg)

        nvidia_sensitive_patterns = [
            'nv_api_key',
            'nvidia_api_key',
            'cuda_api_key',
        ]

        for attr in nvidia_sensitive_patterns:
            pattern = rf"{attr}='?([\w-]+)'?"
            msg = re.sub(pattern, f"{attr}='******'", msg)

        # Also check for NV_API in environment variable names
        for key, value in os.environ.items():
            if 'NV_API' in key.upper() and len(value) > 2 and value != 'default':
                msg = msg.replace(value, '******')

        record.msg = msg
        record.args = ()
        return result


def get_nv_console_handler(log_level: int = logging.INFO) -> logging.StreamHandler:
    """Returns a console handler for NVIDIA logging with cyan color accent."""
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter_str = '\033[96m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s'
    console_handler.setFormatter(NvColoredFormatter(formatter_str, datefmt='%H:%M:%S'))
    return console_handler


def get_nv_file_handler(
    log_dir: str, log_level: int = logging.INFO
) -> logging.FileHandler:
    """Returns a file handler for NVIDIA logging with nvidia_ prefix."""
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y-%m-%d')
    file_name = f'nvidia_{timestamp}.log'
    file_handler = logging.FileHandler(os.path.join(log_dir, file_name))
    file_handler.setLevel(log_level)
    if NV_LOG_JSON:
        file_handler.setFormatter(json_formatter())
    else:
        file_handler.setFormatter(
            NoColorFormatter(
                '%(asctime)s - %(name)s:%(levelname)s: %(filename)s:%(lineno)s - %(message)s',
                datefmt='%H:%M:%S',
            )
        )
    return file_handler


# Create NVIDIA logger
nvidia_logger = logging.getLogger('nvidia')
nv_current_log_level = logging.INFO

if NV_LOG_LEVEL in logging.getLevelNamesMapping():
    nv_current_log_level = logging.getLevelNamesMapping()[NV_LOG_LEVEL]
nvidia_logger.setLevel(nv_current_log_level)

if NV_DEBUG:
    nvidia_logger.addFilter(StackInfoFilter())

if nv_current_log_level == logging.DEBUG:
    NV_LOG_TO_FILE = True
    nvidia_logger.debug('NV_DEBUG mode enabled.')

if NV_LOG_JSON:
    nvidia_logger.addHandler(json_log_handler(nv_current_log_level))
else:
    nvidia_logger.addHandler(get_nv_console_handler(nv_current_log_level))

nvidia_logger.addFilter(NvSensitiveDataFilter())
nvidia_logger.propagate = False
nvidia_logger.debug('NVIDIA Logging initialized')

# NVIDIA-specific log directory
NV_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'logs',
    'nvidia',
)

if NV_LOG_TO_FILE:
    nvidia_logger.addHandler(get_nv_file_handler(NV_LOG_DIR, nv_current_log_level))
    nvidia_logger.debug(f'NVIDIA Logging to file in: {NV_LOG_DIR}')

# Suppress NVIDIA-specific verbose loggers
NV_LOQUACIOUS_LOGGERS = [
    'pynvml',
    'nvidia-ml-py',
    'cupy',
    'numba',
]

for logger_name in NV_LOQUACIOUS_LOGGERS:
    logging.getLogger(logger_name).setLevel('WARNING')


class NvLlmFileHandler(LlmFileHandler):
    """NVIDIA LLM file handler that logs to nvidia subdirectory."""

    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        encoding: str = 'utf-8',
        delay: bool = False,
    ):
        self.filename = filename
        self.message_counter = 1
        if NV_DEBUG:
            from datetime import datetime

            self.session = datetime.now().strftime('%y-%m-%d_%H-%M')
        else:
            self.session = 'default'
        self.log_directory = os.path.join(NV_LOG_DIR, 'llm', self.session)
        os.makedirs(self.log_directory, exist_ok=True)
        if not NV_DEBUG:
            # Clear the log directory if not in debug mode
            for file in os.listdir(self.log_directory):
                file_path = os.path.join(self.log_directory, file)
                try:
                    os.unlink(file_path)
                except Exception as e:
                    nvidia_logger.error('Failed to delete %s. Reason: %s', file_path, e)
        filename = f'{self.filename}_{self.message_counter:03}.log'
        self.baseFilename = os.path.join(self.log_directory, filename)
        logging.FileHandler.__init__(self, self.baseFilename, mode, encoding, delay)

    def emit(self, record: logging.LogRecord) -> None:
        filename = f'{self.filename}_{self.message_counter:03}.log'
        self.baseFilename = os.path.join(self.log_directory, filename)
        self.stream = self._open()
        logging.FileHandler.emit(self, record)
        self.stream.close()
        nvidia_logger.debug('NVIDIA Logging to %s', self.baseFilename)
        self.message_counter += 1


def _setup_nv_llm_logger(name: str, log_level: int) -> logging.Logger:
    """Setup NVIDIA LLM logger with nvidia prefix."""
    logger = logging.getLogger(f'nvidia.{name}')
    logger.propagate = False
    logger.setLevel(log_level)
    if NV_LOG_TO_FILE:
        nv_llm_file_handler = NvLlmFileHandler(name, delay=True)
        nv_llm_file_handler.setFormatter(logging.Formatter('%(message)s'))
        nv_llm_file_handler.setLevel(log_level)
        logger.addHandler(nv_llm_file_handler)
    return logger


# Create NVIDIA LLM loggers
nv_llm_prompt_logger = _setup_nv_llm_logger('prompt', nv_current_log_level)
nv_llm_response_logger = _setup_nv_llm_logger('response', nv_current_log_level)


class NvidiaLoggerAdapter(OpenHandsLoggerAdapter):
    """NVIDIA logger adapter that uses nvidia_logger as default."""

    def __init__(
        self, logger: logging.Logger = nvidia_logger, extra: dict | None = None
    ):
        super().__init__(logger, extra)
