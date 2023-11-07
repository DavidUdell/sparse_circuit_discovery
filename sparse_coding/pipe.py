"""Run the full sparse coding pipeline in one command."""


from runpy import run_module


for script in [
    "collect_acts",
    "train_autoencoder",
    "interp_tools.top_tokens",
]:
    try:
        run_module(f"sparse_coding.{script}")
    except Exception as e:  # pylint: disable=broad-except
        print(f"Fatal error at script {script}: {e}")
