import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration management for the ML project"""
    
    def __init__(self):
        self.model_config = self._load_yaml('config/model_config.yaml')
        self.api_config = self._load_yaml('config/api_config.yaml')
        
    def _load_yaml(self, filepath: str) -> Dict[str, Any]:
        try:
            with open(filepath, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Warning: Config file {filepath} not found. Using defaults.")
            return {}
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config
    
    def get_api_config(self) -> Dict[str, Any]:
        return self.api_config
config = Config()