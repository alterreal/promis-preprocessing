"""
Configuration loader for YAML-based configuration.
"""
import yaml
from pathlib import Path


def load_config(config_path=None):
    """Load YAML configuration file."""
    if config_path is None:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        config_path = project_root / "config.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Load and export the configuration
config = load_config()
