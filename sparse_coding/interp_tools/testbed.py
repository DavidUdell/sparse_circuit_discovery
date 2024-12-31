"""A small demonstration of the basic structure of grad_graph.py"""

from contextlib import contextmanager

import torch as t


class SimpleModel(t.nn.Module):
    """Toy single layer model."""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = t.nn.Linear(3, 3)

    def forward(self, x):
        """Linear-only forward pass."""
        return self.linear(x)


model = SimpleModel()
input_tensor = t.randn(1, 3, requires_grad=True)
target = t.tensor([[1.0, 0.0, 0.0]])

loss_fn = t.nn.MSELoss()

projection = t.nn.Linear(3, 6)
reconstruction = t.nn.Linear(6, 3)

grads_dict = {}


@contextmanager
def hooks_manager(model_in, projection_in, reconstruction_in):
    """Demonstration forward-then-backward hooks context manager."""
    gradients = {}
    handles = []

    def backward_hook(grad):
        gradients["grads"] = grad

    def forward_hook(
        module, inputs, output  # pylint: disable=unused-argument
    ):
        output = projection_in(output)
        output = t.relu(output)
        handles.append(output.register_hook(backward_hook))
        output = reconstruction_in(output)
        return output

    handles.append(model_in.register_forward_hook(forward_hook))

    try:
        yield gradients
    finally:
        for h in handles:
            h.remove()


with hooks_manager(model, projection, reconstruction) as grads:
    activation = model(input_tensor)
    loss = loss_fn(activation, target)
    loss.backward(retain_graph=True)
    grads_dict = grads
    print(grads_dict["grads"], end="\n\n")

    intermediate_scalar = activation[:, 1]
    intermediate_scalar.backward(retain_graph=True)
    grads_dict = grads
    print(grads_dict["grads"], end="\n\n")

    intermediate_scalar = activation[:, -1]
    intermediate_scalar.backward(retain_graph=True)
    grads_dict = grads
    print(grads_dict["grads"], end="\n\n")
