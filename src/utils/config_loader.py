import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigLoader:
    """
    Utility class to load and manage YAML configuration files for the ScienceAssist pipeline.
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a specific configuration file.
        
        Args:
            config_name: Name of the config file (without .yaml extension)
            
        Returns:
            Dictionary containing the configuration
        """
        config_file = self.config_dir / f"{config_name}_config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        try:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
            self.logger.info(f"Loaded configuration from {config_file}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_file}: {e}")
    
    def load_main_config(self) -> Dict[str, Any]:
        """Load the main configuration file."""
        return self.load_config("main")
    
    def get_pipeline_config(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Load configuration for a specific pipeline.
        
        Args:
            pipeline_name: Name of the pipeline (data_download, data_extraction, etc.)
            
        Returns:
            Dictionary containing the pipeline configuration
        """
        return self.load_config(pipeline_name)
    
    def get_nested_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        Get a nested value from configuration using dot notation.
        
        Args:
            config: Configuration dictionary
            key_path: Path to the value (e.g., "paths.downloads_dir")
            default: Default value if key doesn't exist
            
        Returns:
            The value at the specified path
        """
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def validate_config(self, config: Dict[str, Any], required_keys: list) -> bool:
        """
        Validate that required keys exist in the configuration.
        
        Args:
            config: Configuration dictionary
            required_keys: List of required keys (can use dot notation)
            
        Returns:
            True if all required keys exist
        """
        for key in required_keys:
            if self.get_nested_value(config, key) is None:
                self.logger.error(f"Missing required configuration key: {key}")
                return False
        return True

# Convenience functions for common config operations from main config folder
def load_pipeline_config(pipeline_name: str) -> Dict[str, Any]:
    """Load configuration for a specific pipeline."""
    loader = ConfigLoader()
    return loader.get_pipeline_config(pipeline_name)

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get a value from configuration using dot notation."""
    loader = ConfigLoader()
    return loader.get_nested_value(config, key_path, default) 