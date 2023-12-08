FROM pytorch/pytorch:latest

# To suppress warning.
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS="yes"

# Install git and graphviz; gcc is a secret dependency of graphviz.
RUN apt-get update && apt-get install -y git gcc graphviz graphviz-dev

# Install repository.
WORKDIR /root/sparse_circuit_discovery
COPY . .
RUN pip install --no-cache-dir --editable .

# Entrypoint for bash.
ENTRYPOINT ["/bin/bash", "-il"]
