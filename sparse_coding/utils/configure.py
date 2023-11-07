"""Function for loading YAML config files."""


from pathlib import Path
from textwrap import dedent

import yaml


def load_yaml_constants(base_file):
    """Load config files with get() methods."""

    current_dir = Path(base_file).parent
    if current_dir.name == "sparse_coding":
        hf_access_path = current_dir / "config/hf_access.yaml"
        central_config_path = current_dir / "config/central_config.yaml"
    elif current_dir.name == "interp_scripts":
        hf_access_path = current_dir.parent / "config/hf_access.yaml"
        central_config_path = current_dir.parent / "config/central_config.yaml"
    else:
        raise ValueError(
            dedent(
                f"""
                Trying to access config files from an unfamiliar present
                directory: {current_dir}
                """
            )
        )

    try:
        with open(hf_access_path, "r", encoding="utf-8") as f:
            access = yaml.safe_load(f)
    except FileNotFoundError:
        print("hf_access.yaml not found. Creating it now.")
        with open(hf_access_path, "w", encoding="utf-8") as w:
            w.write('HF_ACCESS_TOKEN: ""\n')
        access = {}
    except yaml.YAMLError as e:
        print(e)

    with open(central_config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

    return access, config
