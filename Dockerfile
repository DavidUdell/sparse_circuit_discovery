FROM pytorch/pytorch:latest

# Install git.
RUN apt-get update && apt-get install -y git

# Install repository.
WORKDIR /root/sparse_circuit_discovery
COPY . .
RUN pip install --no-cache-dir --editable .

# Entrypoint for bash.
ENTRYPOINT ["/bin/bash", "-il"]
