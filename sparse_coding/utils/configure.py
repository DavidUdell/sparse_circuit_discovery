"""Function for loading YAML config files."""


from pathlib import Path
from textwrap import dedent

import yaml


def load_yaml_constants(base_file):
    """Load config files."""

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
                Trying to access config files from an unfamiliar working
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


def save_paths(base_file, save_append: str) -> str:
    """Route to save paths from the current working directory."""

    assert isinstance(
        save_append, str
    ), f"`save_append` must be a string: {save_append}."

    current_dir = Path(base_file).parent

    if current_dir.name == "sparse_coding":
        save_path = current_dir / "data" / save_append
        return str(save_path)

    if current_dir.name == "interp_scripts":
        save_path = current_dir.parent / "data" / save_append
        return str(save_path)

    raise ValueError(
        dedent(
            f"""
            Trying to route to save directory from an unfamiliar working
            directory:
            {current_dir}
            """
        )
    )
