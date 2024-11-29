import os
import yaml
import logging

class ConfigManager:
    def __init__(self, config_path='config.yaml'):
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            logging.error("Configuration file not found.")
            raise
        except yaml.YAMLError:
            logging.error("Error parsing YAML file.")
            raise

    def get_config(self, key):
        return self.config.get(key)