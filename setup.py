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
    # pylint:disable=line-too-long
    install_requires=[
        "accelerate==1.5.2",
        "datasets==2.21.0",
        "einops==0.8.1",
        "jupyter==1.1.1",
        "lightning==2.5.1",
        "matplotlib==3.10.1",
        "nnsight==0.4.5",
        "numpy==2.2.4",
        "nvidia-ml-py==12.570.86",
        "pygraphviz==1.14",
        "pytest==8.3.5",
        "PyYAML==6.0.1",
        "scikit-learn==1.6.1",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "tqdm==4.67.1",
        "tracr @ git+https://github.com/google-deepmind/tracr.git@9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd",
        "transformer-lens==2.15.0",
        "transformers==4.49.0",
        "wandb==0.19.8",
    ],
)
