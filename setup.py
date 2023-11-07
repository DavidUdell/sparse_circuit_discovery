"""Install dependencies for `sparse_circuit_discovery`."""


from setuptools import setup, find_packages


setup(
    name="sparse_circuit_discovery",
    description="Automatic circuit discovery in LLMs, using sparse coding.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="David Udell",
    version="0.1.2",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "jupyter",
        "numpy",
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "lightning",
        "scikit-learn",
        "PyYAML",
        "pytest",
        "circuitsvis",
        "prettytable",
        "openai",
    ],
)
