"""Run the main sparse coding pipeline in one command."""


from runpy import run_module


for script in [
    "collect_acts_webtext",
    "train_autoencoder",
    "interp_tools.top_tokens",
    "interp_tools.feature_web_mc",
]:
    try:
        run_module(f"sparse_coding.{script}")
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error at script {script}: {e}")
