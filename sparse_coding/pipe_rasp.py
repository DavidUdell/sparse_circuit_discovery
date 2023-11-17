"""Run the rasp validation pipeline in one command."""


from runpy import run_module


for script in [
    "rasp.rasp_cache",
    "interp_tools.feature_web",
]:
    try:
        run_module(f"sparse_coding.{script}")
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error at script {script}: {e}")
