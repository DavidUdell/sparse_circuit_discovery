"""
Smoke integration test for `sparse_coding.py`.

Note that this integration test will necessarily be somewhat slow.
"""


import os
from runpy import run_module

import pytest
import yaml
import torch as t

from sparse_coding.utils.interface import sanitize_model_name


@pytest.fixture
def mock_interface(monkeypatch):
    """Load to and from the smoke test yaml files."""

    def mock_load_yaml_constants(base_file):  # pylint: disable=unused-argument
        """Load config files."""

        try:
            with open(
                "smoke_test_config/smoke_test_access.yaml",
                "r",
                encoding="utf-8",
            ) as f:
                access = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

        with open(
            "smoke_test_config/smoke_test_config.yaml",
            "r",
            encoding="utf-8",
        ) as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)

        return access, config

    monkeypatch.setattr(
        "sparse_coding.utils.interface.load_yaml_constants",
        mock_load_yaml_constants,
    )

    def mock_save_paths(  # pylint: disable=unused-argument
        base_file, save_append
    ) -> str:
        """Route to smoke test save paths from the tests directory."""
        assert isinstance(
            save_append, str
        ), f"`save_append` must be a string: {save_append}."
        return "smoke_test_data/" + save_append

    monkeypatch.setattr(
        "sparse_coding.utils.interface.save_paths", mock_save_paths
    )

    def mock_cache_layer_tensor(  # pylint: disable=unused-argument
        layer_tensor,
        layer_idx: int,
        save_append: str,
        base_file: str,
        model_name: str,
    ) -> None:
        """Forcibly cache layer tensors in the smoke test directory."""

        assert isinstance(
            layer_idx, int
        ), f"Layer index {layer_idx} is not an int."

        assert not isinstance(
            layer_idx, bool
        ), f"Layer index {layer_idx} is a bool, not an int."

        safe_model_name = sanitize_model_name(model_name)
        save_subdir_path = (
            "smoke_test_data/" + f"/{safe_model_name}/{layer_idx}"
        )

        os.makedirs(save_subdir_path, exist_ok=True)
        t.save(layer_tensor, save_subdir_path + f"/{save_append}")

    monkeypatch.setattr(
        "sparse_coding.utils.interface.cache_layer_tensor",
        mock_cache_layer_tensor,
    )


def test_smoke_sparse_coding(
    mock_interface,
):  # pylint: disable=redefined-outer-name, unused-argument
    """Run the submodule scripts in sequence."""
    for script in [
        "collect_acts_mc",
        "train_autoencoder",
        "interp_tools.top_tokens",
        "interp_tools.feature_web",
    ]:
        try:
            run_module(f"sparse_coding.{script}")
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail(f"Smoke test for {script} failed: {e}")
