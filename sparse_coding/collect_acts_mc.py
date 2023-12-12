# %%
"""
Collects model activations while running Truthful-QA multiple-choice evals.

An implementation of the Truthful-QA multiple-choice task. I'm interested in
collecting residual activations during TruthfulQA to train a variational
autoencoder on, for the purpose of finding task-relevant activation directions
in the model's residual space. The script will collect those activation tensors
and their prompts and save them to disk during the eval. Requires a HuggingFace
access token for the `Llama-2` models.
"""


import warnings

import numpy as np
import torch as t
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from sparse_coding.utils.interface import (
    parse_slice,
    validate_slice,
    cache_layer_tensor,
    slice_to_seq,
    load_yaml_constants,
    save_paths,
    pad_activations,
)
from sparse_coding.utils.tasks import multiple_choice_task


assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B requires at least transformers v4.31.0"

# %%
# Set up constants.
access, config = load_yaml_constants(__file__)

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
MODEL_DIR = config.get("MODEL_DIR")
PROMPT_IDS_PATH = save_paths(__file__, config.get("PROMPT_IDS_FILE"))
ACTS_DATA_FILE = config.get("ACTS_DATA_FILE")
ACTS_LAYERS_SLICE = parse_slice(config.get("ACTS_LAYERS_SLICE"))
SEED = config.get("SEED")
MAX_NEW_TOKENS = config.get("MAX_NEW_TOKENS", 1)
NUM_RETURN_SEQUENCES = config.get("NUM_RETURN_SEQUENCES", 1)
NUM_SHOT = config.get("NUM_SHOT", 6)
NUM_QUESTIONS_EVALED = config.get("NUM_QUESTIONS_EVALED", 800)

assert (
    NUM_QUESTIONS_EVALED > NUM_SHOT
), "There must be a question not used for the multishot demonstration."

# %%
# Reproducibility.
_ = t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Efficient inference and model parallelization.
t.set_grad_enabled(False)
accelerator: Accelerator = Accelerator()
# `device_map="auto` helps initialize big models. Note that the HF transformers
# `CausalLMOutput` class has both a `hidden_states` and an `attentions`
# attribute, so most of the internal tensors we need should be available
# through the vanilla HF API. We'll have to write and register out own hook
# factory to extract MLP output activations, though; for whatever reason, the
# HF API doesn't expose those to us.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        token=HF_ACCESS_TOKEN,
        output_hidden_states=True,
    )
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    token=HF_ACCESS_TOKEN,
)
model: PreTrainedModel = accelerator.prepare(model)
model.eval()

# Validate slice against model's layer count.
validate_slice(model, ACTS_LAYERS_SLICE)

# %%
# Load the TruthfulQA dataset.
dataset: dict = load_dataset("truthful_qa", "multiple_choice")

assert (
    len(dataset["validation"]["question"]) >= NUM_QUESTIONS_EVALED
), "More datapoints sampled than exist in the dataset."

all_indices: np.ndarray = np.random.choice(
    len(dataset["validation"]["question"]),
    size=len(dataset["validation"]["question"]),
    replace=False,
)
sampled_indices: list = all_indices[:NUM_QUESTIONS_EVALED].tolist()

# %%
# Collect activations.
activations, answers_with_rubric, prompts_ids = multiple_choice_task(
    dataset,
    sampled_indices,
    model,
    tokenizer,
    accelerator,
    NUM_SHOT,
    ACTS_LAYERS_SLICE,
)

# %%
# Grade the model's answers.
model_accuracy: float = 0.0
for (
    question_idx
) in answers_with_rubric:  # pylint: disable=consider-using-dict-items
    if (
        answers_with_rubric[question_idx][0]
        == answers_with_rubric[question_idx][1]
    ):
        model_accuracy += 1.0

model_accuracy /= len(answers_with_rubric)
print(f"{MODEL_DIR} accuracy:{round(model_accuracy*100, 2)}%.")


# %%
# Save prompt ids.
prompt_ids_list: list = []
for question_ids in prompts_ids:
    prompt_ids_list.append(question_ids.tolist())
prompt_ids_array: np.ndarray = np.array(prompt_ids_list, dtype=object)
np.save(PROMPT_IDS_PATH, prompt_ids_array, allow_pickle=True)
# array of (x, 1)
# Each element along x is a list of ints, of seq len.

# %%
# Save activations.
seq_layer_indices: range = slice_to_seq(model, ACTS_LAYERS_SLICE)

# Deal with the single layer case, since there are no tuples there.
if isinstance(activations, list) and isinstance(activations[0], t.Tensor):
    activations: list[tuple[t.Tensor]] = [(tensor,) for tensor in activations]

max_seq_len: int = max(
    tensor.size(1) for layers_tuple in activations for tensor in layers_tuple
)

for abs_idx, layer_idx in enumerate(seq_layer_indices):
    # Pad activations to the widest activation stream-dim.
    padded_activations: list[t.Tensor] = [
        pad_activations(layers_tuple[abs_idx], max_seq_len, accelerator)
        for layers_tuple in activations
    ]
    concat_activations: t.Tensor = t.cat(
        padded_activations,
        dim=0,
    )

    cache_layer_tensor(
        concat_activations,
        layer_idx,
        ACTS_DATA_FILE,
        __file__,
        MODEL_DIR,
    )
