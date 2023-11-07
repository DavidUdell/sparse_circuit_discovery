"""
Smoke integration test for `sparse_coding.py`.

Note that this integration test will necessarily be somewhat slow.
"""


from runpy import run_module

import pytest
import yaml


@pytest.fixture
def mock_configure(monkeypatch):
    """Load to and from the smoke test yaml files."""

    def mock_load_yaml_constants(base_file):  # pylint: disable=unused-argument
        """Load config files."""

        try:
            with open(
            "smoke_test_config/smoke_test_access.yaml",
            "r",
            encoding="utf-8"
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

    def mock_save_paths(  # pylint: disable=unused-argument
        base_file, save_append
    ) -> str:
        """Route to smoke test save paths from the tests directory."""
        assert isinstance(
            save_append, str
        ), f"`save_append` must be a string: {save_append}."
        return "smoke_test_data/" + save_append

    monkeypatch.setattr(
        "sparse_coding.utils.configure.load_yaml_constants",
        mock_load_yaml_constants,
    )
    monkeypatch.setattr(
        "sparse_coding.utils.configure.save_paths", mock_save_paths
    )


def test_smoke_sparse_coding(
    mock_configure,
):  # pylint: disable=redefined-outer-name, unused-argument
    """Run the submodule scripts in sequence."""
    for script in [
        "collect_acts",
        "train_autoencoder",
        "interp_tools.top_tokens",
    ]:
        try:
            run_module(f"sparse_coding.{script}")
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail(f"Smoke test for {script} failed: {e}")
