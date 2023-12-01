"""Functions for running model inference tasks."""


import numpy as np
import torch as t
from tqdm.auto import tqdm


def shuffle_answers(choices, labels_one_hot):
    """Shuffle the answers and the answer labels correspondingly."""

    paired_choices = list(zip(choices, labels_one_hot))
    np.random.shuffle(paired_choices)
    choices, labels_one_hot = zip(*paired_choices)

    return choices, labels_one_hot


def unhot(labels) -> int:
    """Change the one-hot ground truth labels to a 1-indexed int."""

    return np.argmax(labels) + 1


def multiple_choice_task(
    dataset,
    indices,
    model,
    tokenizer,
    accelerator,
    num_shot: int,
    acts_layers_slice: slice,
    streamlined_mode: bool = False,
) -> tuple[list, dict, list] | None:
    """
    The model does the `truthful_qa multiple-choice 1` task.
    
    `streamlined_mode` runs the task without returning any outputs.
    """

    activations = []
    answers_with_rubric = {}
    prompts_ids = []

    for question_idx in tqdm(indices, desc="Questions Progress", leave=False):
        multishot = ""
        # The multishot part of the prompt should not include the current
        # question.
        multishot_indices: np.ndarray = np.random.choice(
            [
                x
                for x in range(len(dataset["validation"]["question"]))
                if x != question_idx
            ],
            size=num_shot,
            replace=False,
        )
        for multishot_idx in multishot_indices:
            multishot += (
                "Q: " + dataset["validation"]["question"][multishot_idx] + "\n"
            )
            unshuffled_choices: list = dataset["validation"]["mc1_targets"][
                multishot_idx
            ]["choices"]
            unshuffled_labels: list = dataset["validation"]["mc1_targets"][
                multishot_idx
            ]["labels"]
            shuffled_choices, shuffled_labels = shuffle_answers(
                unshuffled_choices, unshuffled_labels
            )

            for choice_num, shuffled_choice in enumerate(shuffled_choices):
                # `choice_num` is 0-indexed; I want 1-indexed m/c choices.
                multishot += (
                    "(" + str(choice_num + 1) + ") " + shuffled_choice + "\n"
                )

            correct_answer: int = unhot(shuffled_labels)
            multishot += "A: (" + str(correct_answer) + ")\n"

        # Concat the current question with shuffled choices to the prompt.
        question: str = (
            "Q: " + dataset["validation"]["question"][question_idx] + "\n"
        )
        unshuffled_choices_current: list = dataset["validation"][
            "mc1_targets"
        ][question_idx]["choices"]
        unshuffled_labels_current: list = dataset["validation"]["mc1_targets"][
            question_idx
        ]["labels"]
        shuffled_choices_current, shuffled_labels_current = shuffle_answers(
            unshuffled_choices_current, unshuffled_labels_current
        )
        for option_num, shuffled_option in enumerate(shuffled_choices_current):
            # `option_num` is 0-indexed; I want 1-indexed m/c choices.
            question += (
                "(" + str(option_num + 1) + ") " + shuffled_option + "\n"
            )
        # Opening paren, for just a multiple-choice answer integer.
        question += "A: ("

        input_ids: t.Tensor = tokenizer.encode(
            multishot + question, return_tensors="pt"
        )
        prompts_ids.append(input_ids)

        # `accelerate` parallelization can fail with small models.
        try:
            input_ids = accelerator.prepare(input_ids)
            outputs = model(input_ids)
        except RuntimeError:
            input_ids = input_ids.to(model.device)
            input_ids = accelerator.prepare(input_ids)
            outputs = model(input_ids)

        if streamlined_mode:
            continue
        # We want the answer stream's logits, so we pass
        # `outputs.logits[:,-1,:]`. `dim=-1` means greedy sampling over the
        # token dim.
        answer_id: t.LongTensor = t.argmax(outputs.logits[:, -1, :], dim=-1)
        model_answer: str = tokenizer.decode(answer_id)

        # Cut the completion down to just its answer integer.
        model_answer = model_answer.split("\n")[-1]
        model_answer = model_answer.replace("A: (", "")

        ground_truth: int = unhot(shuffled_labels_current)
        answers_with_rubric[question_idx] = [int(model_answer), ground_truth]
        activations.append(outputs.hidden_states[acts_layers_slice])

    return activations, answers_with_rubric, prompts_ids
