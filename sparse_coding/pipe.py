"""Run the main sparse coding pipeline in one command."""

from textwrap import dedent

from runpy import run_module


print(
    dedent(
        """
        For the time being,
        1. `model_dir` must be `openai-community/gpt2`,
        2. `projection_factor` must be 32,
        3. autoencoder interp data has been precomputed, and
        4. only ablation studies are performed and measured.
        """
    )
)

# "collect_acts",
# "interp_tools.contexts",
for script in [
    "load_autoencoder",
    "interp_tools.cognition_graph_webtext",
]:
    try:
        run_module(f"sparse_coding.{script}")
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error at script {script}: {e}")
