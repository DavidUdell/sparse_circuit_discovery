FROM pytorch/pytorch:latest

# Set up conda.
RUN conda update --name base --channel defaults --yes conda && \
    conda init bash && \
    conda create --name interp && \
    echo "conda activate interp" >> ~/.bashrc && \
    conda install --name interp --channel conda-forge --yes git

# Install repo and dependencies
WORKDIR /root/sparse_circuit_discovery
COPY . .
RUN conda run --name interp pip install --no-cache-dir --editable .

# Entrypoint for bash.
ENTRYPOINT ["/bin/bash", "-il"]
