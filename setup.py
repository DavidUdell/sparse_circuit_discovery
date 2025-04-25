"""Install dependencies for `sparse_circuit_discovery`."""

from setuptools import setup, find_packages


setup(
    name="sparse_circuit_discovery",
    description="Circuit discovery in GPT-2 small, using sparse autocoding",
    long_description=open("README.markdown", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="David Udell",
    version="1.5.0",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "accelerate",
        "datasets==2.21.0",
        "einops",
        "jupyter",
        "lightning",
        "matplotlib",
        "nnsight",
        "numpy>=2.2.2",
        "nvidia-ml-py",
        "pygraphviz",
        "pytest",
        "PyYAML",
        "scikit-learn",
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "tqdm",
        "tracr @ git+https://github.com/google-deepmind/tracr.git",
        "transformer_lens",
        "transformers",
        "wandb",
    ],
)
