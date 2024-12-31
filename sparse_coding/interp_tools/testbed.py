"""A small demonstration of the basic structure of grad_graph.py"""

from contextlib import contextmanager

import torch as t


class SimpleModel(t.nn.Module):
    """Toy two-layer model."""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear_1 = t.nn.Linear(3, 3)
        self.linear_2 = t.nn.Linear(3, 3)

    def forward(self, x):
        """Linear-only forward pass."""
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x


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

    def backward_replace_fac(replace: t.Tensor):
        def replace_hook(grad):  # pylint: disable=unused-argument
            return replace.grad

        return replace_hook

    def backward_hook_fac(name: str):
        def backward_hook(grad):
            gradients[name] = grad

        return backward_hook

    def forward_hook_fac(name: str):
        def forward_hook(
            module, inputs, output  # pylint: disable=unused-argument
        ):
            projected_acts = projection_in(output)
            projected_acts = t.relu(projected_acts)
            handles.append(
                projected_acts.register_hook(backward_hook_fac(name))
            )
            decoded_acts = reconstruction_in(projected_acts)
            error = output - decoded_acts
            error = error.detach().requires_grad_(True)
            reconstructed = decoded_acts + error
            reconstructed.retain_grad()
            handles.append(
                output.register_hook(backward_replace_fac(reconstructed))
            )
            return reconstructed

        return forward_hook

    handles.append(
        model_in.linear_1.register_forward_hook(forward_hook_fac("Linear_1"))
    )
    handles.append(
        model_in.linear_2.register_forward_hook(forward_hook_fac("Linear_2"))
    )

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
    print(grads_dict["Linear_1"])
    print(grads_dict["Linear_2"], end="\n\n")

    intermediate_scalar = activation[:, 1]
    intermediate_scalar.backward(retain_graph=True)
    grads_dict = grads
    print(grads_dict["Linear_1"])
    print(grads_dict["Linear_2"], end="\n\n")

    intermediate_scalar = activation[:, -1]
    intermediate_scalar.backward(retain_graph=True)
    grads_dict = grads
    print(grads_dict["Linear_1"])
    print(grads_dict["Linear_2"], end="\n\n")
