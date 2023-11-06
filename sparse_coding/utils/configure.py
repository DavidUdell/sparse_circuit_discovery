"""Function for loading YAML config files."""
import yaml


def load_yaml_constants():
    """Load config files with get() methods."""

    try:
        with open(
            "/root/sparse_circuit_discovery/sparse_coding/hf_access.yaml",
            "r",
            encoding="utf-8",
        ) as f:
            access = yaml.safe_load(f)
    except FileNotFoundError:
        print("hf_access.yaml not found. Creating it now.")
        with open(
            "/root/sparse_circuit_discovery/sparse_coding/hf_access.yaml",
            "w",
            encoding="utf-8",
        ) as w:
            w.write('HF_ACCESS_TOKEN: ""\n')
        access = {}
    except yaml.YAMLError as e:
        print(e)
    with open(
        "/root/sparse_circuit_discovery/sparse_coding/central_config.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

    return access, config
