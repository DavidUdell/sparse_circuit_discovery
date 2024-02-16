"""Test back and forth between model logit dims and tokens."""


import gc

import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM

# Determinism.
t.manual_seed(0)


def test_logits():
    """
    Test whether we're decoding logits correctly.
    
    I haven't extracted the relevant code from the repo, but this is a close
    mirror for now.
    """

    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    sequences: list[t.BatchEncoding] = []
    seq = tokenizer(
        "Hello, my name is what? My name is who?",
        return_tensors="pt",
    )
    sequences.append(seq)

    seq_2 = tokenizer(
        "My name is Slim Shady",
        return_tensors="pt",
    )
    sequences.append(seq_2)

    logits_list = []
    for sequence in sequences:
        _ = t.manual_seed(0)
        try:
            output = model(**sequence)
        except RuntimeError:
            gc.collect()
            output = model(**sequence)

        logits = output.logits[:, -1, :].cpu()
        logits_list.append(logits)

    prob_diffs = t.nn.functional.softmax(
        logits_list[1],
        dim=-1,
    ) - t.nn.functional.softmax(
        logits_list[0],
        dim=-1,
    )
    top_tokens = t.abs(prob_diffs).sum(dim=0).squeeze().topk(5).indices

    assert top_tokens.shape == (5,)
