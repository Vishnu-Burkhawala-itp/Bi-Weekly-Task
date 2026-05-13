import yaml
from typing import Dict, Any

def convert_dict_to_yaml(data: Dict[str, Any]) -> str:
    """
    Enterprise Secret: Converts messy JSON/Dictionaries into clean YAML.
    sort_keys=False ensures the AI reads 'Title' before 'Description', 
    saving massive amounts of tokens and boosting Rerank accuracy.
    """
    return yaml.dump(data, sort_keys=False, default_flow_style=False)