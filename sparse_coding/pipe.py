"""Run the main sparse coding pipeline in one command."""

from textwrap import dedent

from runpy import run_module


print(
    dedent(
        """
        For the time being, model_dir must be openai-community/gpt2 and
        projection_factor must be 32, and only ablations are performed.
        """
    )
)

for script in [
    "collect_acts",
    "load_autoencoder",
    "interp_tools.contexts",
    "interp_tools.directed_graph_webtext",
]:
    try:
        run_module(f"sparse_coding.{script}")
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error at script {script}: {e}")
