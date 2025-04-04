from expkit import (
    ExpSetup,
)
import logging
from expkit.storage import CachedRO, ZipStorage, DiskStorage
import os
import shutil
import numpy as np
import re
import json

from transformers import AutoTokenizer

from qflow.utils.data import get_data_iterable, general_process_data
from qflow.utils.math import get_last_number
from datasets import load_dataset

from qflow.utils.data import FlexiblePromptTemplate


def sample_pairs(positive, negatives, n=1):
    nmin = min(len(positive), len(negatives), n)

    if nmin > 0:
        np.random.shuffle(positive)
        np.random.shuffle(negatives)

        if len(positive) < n:
            positive = np.random.choice(
                positive,
                size=n,
                replace=True,
            )
        if len(negatives) < n:
            negatives = np.random.choice(
                negatives,
                size=n,
                replace=True,
            )

        return [
            {
                "chosen": c,
                "rejected": r,
            }
            for c, r in zip(
                positive[:n],
                negatives[:n],
            )
        ]
    else:
        return []


def fake_synth_data(i, po):

    number = get_last_number(i["input"]["answer"])

    alt_results = [
        np.round(
            np.random.rand() * float(number),
            1,
        )
        for _ in range(len(po))
    ]

    if "." not in number:
        alt_results = np.ceil(alt_results).astype(int)

    no = [
        answer.replace(
            number,
            str(pseudo_result),
        )
        for answer, pseudo_result in zip(po, alt_results)
    ]

    return no


def linearize(setup, oracle_key="lastnumber"):
    dataset = []
    evals = []

    if len(setup) > 1:

        setup = setup.unique("i").sort("i")

    for e in setup:
        try:
            print("Processing", e.meta.get("i", 0))
            print("elements:", len(e.instances()))
            print("evals:", len(e.get_eval(oracle_key)))

            dataset.extend(e.instances())
            evals.extend(e.get_eval(oracle_key))

        except Exception as e:
            logging.error("An error occurred during linearization:", e)
            pass

    return {"data": dataset, "evals": evals}


def remove_tags(text, tags):
    ptext = re.sub("|".join(map(re.escape, tags)), "", text)

    return ptext


def create_positive_and_negative_pairs(dataset_stack, stop_tokens):
    dataset = []
    answers = []

    for i, gt in zip(dataset_stack["data"], dataset_stack["evals"]):

        values = np.array(gt["scores"])
        pind = np.where(values == 1)[0]  # extract index of correct
        nind = np.where(values == 0)[0]  # extract index of incorrect

        po = [
            remove_tags(i["outputs"][j]["text"], stop_tokens) for j in pind
        ]  # get text without special tokens
        no = [
            remove_tags(i["outputs"][j]["text"], stop_tokens) for j in nind
        ]  # get text without special tokens

        all_neg, all_po = (
            len(po) == 0,
            len(no) == 0,
        )

        if all_po:  ## if all positive
            no = fake_synth_data(i, po)  # create fake wrongs .. hopefully this is rare.

        dataset.append((po + [i["input"]["answer"]], no))

    return dataset


def sample_pairs_dataset_chat(
    split_dataset,
    epochs=1,
    dataset_path="openai/gsm8k",
    model_path="allenai/OLMo-7B-0724-Instruct-hf",
    prompt_template="{prompt}",
):
    dataset = []

    for (po, no), i in zip(
        split_dataset,
        get_data_iterable(
            model_path=model_path,
            dataset_path=dataset_path,
            split="train",
            prompt_template=prompt_template,
        ),
    ):

        chat_template = i["chat_template_prompt"]

        po, no = list(set(po)), list(set(no))
        for pair in sample_pairs(
            po,
            no,
            n=epochs,
        ):

            dataset.append(
                {
                    k: chat_template
                    + [
                        {
                            "role": "assistant",
                            "content": v,
                        }
                    ]
                    for k, v in pair.items()
                }
            )

    return dataset


def sample_pairs_dataset_prompt(
    split_dataset,
    epochs=1,
    dataset_path="openai/gsm8k",
    model_path="allenai/OLMo-7B-0724-Instruct-hf",
    use_few_shot=False,
):
    dataset = []

    for (po, no), i in zip(
        split_dataset,
        get_data_iterable(
            model_path=model_path,
            dataset_path=dataset_path,
            split="train",
            format="prompt",
            use_few_shot=use_few_shot,
        ),
    ):

        chat_template = [
            {
                "role": "user",
                "content": i["prompt"],
            }
        ]

        po, no = list(set(po)), list(set(no))
        for pair in sample_pairs(
            po,
            no,
            n=epochs,
        ):

            dataset.append(
                {
                    k: chat_template
                    + [
                        {
                            "role": "assistant",
                            "content": v,
                        }
                    ]
                    for k, v in pair.items()
                }
            )

    return dataset


def create_rm_data_test_pairs(
    model_path,
    dataset_path="openai/gsm8k",
):  # this fake synth data .. is not representative ...
    dataset = []
    # templates = []

    for i in get_data_iterable(model_path=model_path, dataset_path=dataset_path):

        chat_template = i["chat_template_prompt"]

        # templates.append(chat_template)
        number = get_last_number(i["answer"])

        alt_number = np.round(
            np.random.rand() * float(number),
            1,
        )

        if "." not in number:
            alt_number = np.ceil(alt_number).astype(int)

        wrong_answer = i["answer"].replace(
            number,
            str(alt_number),
        )

        answer = i["answer"]

        dataset.append(
            {
                "chosen": chat_template
                + [
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                ],
                "rejected": chat_template
                + [
                    {
                        "role": "assistant",
                        "content": wrong_answer,
                    }
                ],
            }
        )

    return dataset


def convert_llamafactory_format(x):

    return {
        "instruction": x["chosen"][0]["content"],
        "chosen": x["chosen"][-1]["content"],
        "rejected": x["rejected"][-1]["content"],
    }


# /gscratch/ark/graf/rmlearn/data


def get_sharded_setup(
    storage,
    model_path,
    temperature=1.0,
    steps=64,
    dataset="openai/gsm8k",
    oracle_key="lastnumber",
    prompt_template="{prompt}",
):
    setup = (
        ExpSetup(storage).query(
            {
                "steps": steps,
                "temperature": temperature,
                # "prompt_template": "Solve the following grade school math problem step-by-step: {prompt}",
                "model_path": model_path,
                "split": "train",
                "dataset": dataset,
                # "prompt_template": prompt_template,
                # "n": n,
            }
        )
        # .filter(lambda x: "i" in x.meta)
        # .filter(lambda x: x.islocked())
    )

    if len(setup) == 0:
        raise ValueError(
            f"No data found for model {model_path} and dataset {dataset} with steps {steps} and temperature {temperature}"
        )

    setup = setup.filter(lambda x: x.has_eval(oracle_key))
    setup = setup.filter(lambda x: x.get("n") == len(x.instances()))

    if len(setup) == 0:
        raise ValueError(f"No eval data with key:{oracle_key} found.")

    setup = setup.sort("n", reverse=False)

    dataset_stack = linearize(setup, oracle_key=oracle_key)

    return dataset_stack


def file_name(dataset, model_path, steps=128, epochs=1, format="raw", split="train"):
    d = dataset.split("/")[-1].lower()
    m = model_path.split("/")[-1].lower()

    if split == "test":
        return f"{format}_{d}_{m}_{split}.json"
    else:
        return f"{format}_{d}_{m}_{steps}_{epochs}_{split}.json"


def generate_rm_dataset(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    base_dir="outputs/llama3gsm8ktrain/",
    dataset="openai/gsm8k",
    save_path="/gscratch/ark/graf/quest-rlhf/qflow/rm/data/",
    epochs=[1, 2, 4, 8],
    steps=128,
    temperature=1.0,
    stop_tokens=None,
    oracle_key="lastnumber",
    prompt_template="{prompt}",
    prefix="",
):

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if stop_tokens is None:
        stop_tokens = [tokenizer.eos_token]
    else:
        stop_tokens = stop_tokens + [tokenizer.eos_token]

    try:
        storage = ZipStorage(
            base_dir=base_dir,
            mode="r",
        )
    except ValueError as e:
        storage = DiskStorage(
            base_dir=base_dir,
            mode="r",
        )

    storage = CachedRO(storage)

    dataset_stack = get_sharded_setup(
        storage,
        model_path=model_path,
        temperature=temperature,
        steps=steps,
        dataset=dataset,
        oracle_key=oracle_key,
        prompt_template=prompt_template,
    )

    all_pairs = create_positive_and_negative_pairs(
        dataset_stack,
        stop_tokens=stop_tokens,
    )

    for ep in epochs:
        sampled_pairs = sample_pairs_dataset_chat(
            all_pairs,
            epochs=ep,
            model_path=model_path,
            dataset_path=dataset,
            prompt_template=prompt_template,
            # use_few_shot=True,
        )

        sampled_pairs_factory = [convert_llamafactory_format(x) for x in sampled_pairs]

        with open(
            save_path
            + prefix
            + file_name(
                dataset,
                model_path,
                steps,
                ep,
                format="raw",
            ),
            "w",
        ) as fn:
            json.dump(sampled_pairs, fn)

        with open(
            save_path
            + prefix
            + file_name(
                dataset,
                model_path,
                steps,
                ep,
                format="llama-factory",
            ),
            "w",
        ) as f:
            json.dump(sampled_pairs_factory, f)

        print(f"Saved {prefix+file_name(dataset, model_path, steps, ep, format='raw')}")
        print(
            f"Saved {prefix+file_name(dataset, model_path, steps, ep, format='llama-factory')}"
        )


def generate_rm_test_dataset(
    model_path="meta-llama/Llama-3.2-3B-Instruct",
    base_dir="outputs/llama3gsm8ktrain/",
    dataset="openai/gsm8k",
    save_path="/gscratch/ark/graf/quest-rlhf/qflow/rm/data/",
):

    test_dataset = create_rm_data_test_pairs(
        model_path=model_path, dataset_path=dataset
    )

    with open(
        save_path + file_name(dataset, model_path, format="raw", split="test"),
        "w",
    ) as fn:
        json.dump(test_dataset, fn)

    test_data = [convert_llamafactory_format(x) for x in test_dataset]

    with open(
        save_path
        + file_name(dataset, model_path, format="llama-factory", split="test"),
        "w",
    ) as fn:
        json.dump(test_data, fn)


def main(
    model_path="meta-llama/Llama-3.1-8B-Instruct",  # "allenai/Llama-3.1-Tulu-3-8B-SFT",   # "allenai/Llama-3.1-Tulu-3-8B-SFT",  # "meta-llama/Llama-3.1-8B-Instruct",
    dataset="lighteval/MATH",  # "lighteval/MATH",  # "openai/gsm8k",
    save_path="/gscratch/ark/graf/quest-rlhf/qflow/rm/data/",
    base_dir="remote-outputs-llama/",  # "outputs/llama3gsm8ktrain/",
    steps=128,
    prompt_template="Solve the following math problem step-by-step: {prompt}\n\nPresent the answer in LaTex format: \\boxed{Your answer}",  # "Solve the following math problem step-by-step: {prompt}",
    prefix="cotnewextractmath14325",
):  # 90d44980-8ad9-48f2-8c34-99747991f571

    dataset_oracles = {
        "openai/gsm8k": "lastnumber",
        "lighteval/MATH": "lastmath",
    }

    generate_rm_dataset(
        model_path=model_path,
        base_dir=base_dir,
        dataset=dataset,
        save_path=save_path,
        epochs=[1, 2, 4, 8],
        steps=steps,
        temperature=1.0,
        oracle_key=dataset_oracles[dataset],
        prompt_template=FlexiblePromptTemplate(prompt_template),
        prefix=prefix,
    )

    """generate_rm_test_dataset(
        model_path=model_path,
        base_dir=base_dir,
        dataset=dataset,
        save_path=save_path,
    )"""  # this has to improved. We need to generate from the base model for the test set.

    info = {}
    for fn in os.listdir(save_path):

        if "llama-factory" not in fn:
            continue

        key = fn.replace(".json", "")
        info[key] = {
            "file_name": fn,
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        }

    with open(save_path + "dataset_info.json", "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    main()
