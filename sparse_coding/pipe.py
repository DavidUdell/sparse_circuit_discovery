"""Run the main sparse coding pipeline in one command."""

from runpy import run_module


print(
    "For the time being, model_dir must be openai-community/gpt2 and projection_factor must be 32."
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
