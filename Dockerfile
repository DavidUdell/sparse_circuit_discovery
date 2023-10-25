FROM pytorch/pytorch:latest

# Install git.
RUN apt update && apt install -y git

# Install repository.
WORKDIR /root/sparse_circuit_discovery
COPY . .
RUN pip install --no-cache-dir --editable .

# Entrypoint for bash.
ENTRYPOINT ["/bin/bash", "-il"]
