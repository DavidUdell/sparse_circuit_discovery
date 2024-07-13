"""Install dependencies for `sparse_circuit_discovery`."""

from setuptools import setup, find_packages


setup(
    name="sparse_circuit_discovery",
    description="Circuit discovery in GPT-2 small, using sparse autocoding",
    long_description=open("README.markdown", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="David Udell",
    version="1.0.3",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "accelerate",
        "datasets",
        "einops",
        "jupyter",
        "lightning",
        "matplotlib",
        "nnsight",
        "numpy",
        "pygraphviz",
        "pytest",
        "PyYAML",
        "scikit-learn",
        "torch",
        "tqdm",
        "tracr @ git+https://github.com/google-deepmind/tracr.git",
        "transformer_lens",
        "transformers",
        "wandb",
    ],
)
