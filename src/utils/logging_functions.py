import os
import logging
from pathlib import Path
from src.utils.config_loader import get_config_value

def setup_logging(config,pipeline_name):
    """Setup logging based on configuration."""
    if not os.path.isdir(Path(get_config_value(config, 'paths.logs_dir', 'logs'))):
        os.mkdir(Path(get_config_value(config, 'paths.logs_dir', 'logs')))
    log_config = get_config_value(config, 'logging', {})
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_config.get('file', f'logs/{pipeline_name}.log')),
            logging.StreamHandler()
        ]
    )