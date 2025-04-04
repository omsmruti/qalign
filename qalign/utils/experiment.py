from typing import Optional, Dict, Any, List
from datetime import datetime
from tqdm import tqdm
import os
import copy

## expkit
from expkit.exp import Exp
from expkit.storage import DiskStorage

## quest
from quest.reward.model import ContextualRewardModel, ValueHead
from quest.reward.remote import RemoteReward
from quest.proposal import RLHFSuffixProposal
from quest.core import Quest

## qalign
from qalign.utils.data import FlexiblePromptTemplate
from qalign.utils.data import get_data_iterable
from quest.utils.list import chunked


## literegistry
from literegistry import RegistryClient, FileSystemKVStore


def create_experiment(
    save_path: str,
    variant: str,
    model_path: str,
    dataset_path: str,
    n: int,
    temperature: float,
    steps: int,
    max_new_tokens: int = 100,
    max_prompt_length: int = 600,
    batch_size: int = 64,
    split: str = "test",
    prompt_template: str = "{prompt}",
    stop_tokens: Optional[List[str]] = None,
    additional_meta: Optional[Dict[str, Any]] = None,
    format: str = "chat",  # either "chat" or "prompt"
    use_few_shot: bool = False,
) -> Exp:
    """
    Creates a standardized experiment with common metadata.
    """
    if stop_tokens is None:
        stop_tokens = []

    meta = {
        "steps": steps,
        "temperature": temperature,
        "n": n,
        "model_path": model_path,
        "variant": variant,
        "stop_tokens": stop_tokens,
        "max_new_tokens": max_new_tokens,
        "max_prompt_length": max_prompt_length,
        "at": datetime.now().isoformat(),
        "dataset": dataset_path,
        "split": split,
        "prompt_template": prompt_template,
        "batch_size": batch_size,
        "format": format,
        "use_few_shot": use_few_shot,
    }

    if additional_meta:
        meta.update(additional_meta)

    return Exp(
        storage=DiskStorage(save_path, "rw"),
        meta=meta,
    )


def create_extension_experiment(storage, experiment, new_steps=1024):
    samples = [
        {
            "input": (
                data["input"]["input"] if "input" in data["input"] else data["input"]
            ),
            "completion": data["outputs"][-1]["text"],
            "reward": float(data["outputs"][-1]["reward"]),
        }
        for data in experiment.instances(lazy_iterable=True)
    ]

    meta = copy.deepcopy(experiment.meta)

    meta["steps"] = new_steps
    meta["bootstrap"] = samples
    meta["link"] = experiment.get_name()

    new_exp = Exp(
        storage=storage,
        name=experiment.get_name() + "-extension",
        meta=meta,
    )

    return new_exp


def create_vllm_model(
    model_path: str,
    temperature: float,
    max_new_tokens: int = 100,
    max_prompt_length: int = 600,
    stop_tokens: Optional[List[str]] = None,
    device_count: int = 1,
    gpu_memory_utilization: float = 0.8,
    # prompt_template: Optional[str] = None,
    enforce_eager: bool = False,
    remote: bool = False,
):
    """
    Creates a standardized VLLM model instance with common configurations.
    """
    if stop_tokens is None:
        stop_tokens = ["</s>"]

    extra_args = {}
    if "bnb" in model_path:
        extra_args.update(
            {
                "trust_remote_code": True,
                "quantization": "bitsandbytes",
                "load_format": "bitsandbytes",
            }
        )

    model_args = {
        "model_path": model_path,
        "download_dir": os.environ.get("HF_HOME", "/tmp/"),
        "stop_tokens": stop_tokens,
        "temperature": temperature,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": "bfloat16",
        "max_new_tokens": max_new_tokens,
        "max_prompt_length": max_prompt_length,
        "tensor_parallel_size": device_count,
        "enable_prefix_caching": True,
        "enforce_eager": enforce_eager,
        **extra_args,
    }

    if remote:
        from quest.model.remote import RemoteVLLM

        registry = RegistryClient(
            store=FileSystemKVStore("/gscratch/ark/graf/registry"),
            max_history=3600,
            cache_ttl=60,
            service_type="model_path",
        )

        return RemoteVLLM(registry=registry, **model_args)
    else:
        from quest.model.vllm import VLLM

        return VLLM(**model_args)


def process_batch_outputs(
    chain_outputs: Any, batch_size: int
) -> List[List[Dict[str, Any]]]:
    """
    Processes batch outputs from a Quest chain into a standardized format.
    """
    outputs = []
    for i in range(batch_size):
        outputs.append(
            [
                {
                    "t": s["t"],
                    **{k: v[i] for k, v in s.items() if k != "t"},
                }
                for s in chain_outputs.state_path
            ]
        )
    return outputs


def get_batched_data(
    model,
    dataset_path: str,
    split: str,
    n: int,
    batch_size: int,
    prompt_template: str,
    start_index: int = 0,
    num_chains: int = 1,
    completed: int = 0,
    format="chat",
    use_few_shot=False,
) -> List[Any]:
    """
    Gets batched data from a dataset using standard configurations.
    """
    data_iterable = get_data_iterable(
        model_path=model.model_path,
        dataset_path=dataset_path,
        split=split,
        n=start_index + n,
        tokenizer=model.tokenizer,
        prompt_template=FlexiblePromptTemplate(prompt_template),
        format=format,
        use_few_shot=use_few_shot,
    )

    if start_index > 0:
        data_iterable = data_iterable[start_index : start_index + n]

    if num_chains > 1:
        data_iterable = [x for x in data_iterable for _ in range(num_chains)]

    data_iterable = data_iterable[completed:]

    batches = []
    for i in range(0, len(data_iterable), batch_size):
        batches.append(data_iterable[i : i + batch_size])

    return batches


def calculate_reward_scores(
    experiment: Exp,
    reward_key: Optional[str] = None,
) -> List[Dict[str, List[float]]]:
    """
    Calculates reward scores for experiment instances.
    """
    beta = experiment.meta.get("beta", 1.0)

    if not reward_key and "reward_model_path" in experiment.meta:
        reward_key = "crm:" + experiment.meta["reward_model_path"].split(".")[
            0
        ].replace("/", "-")

    return [
        {"scores": [float(o["reward"]) * beta for o in i["outputs"]]}
        for i in experiment.instances(lazy_iterable=True)
    ]


def run_ancestral(experiment, model, steps, data_batches):

    steps = experiment.meta["steps"]

    # Process each batch
    for data_batch in tqdm(data_batches):
        completions_txt = model.ancestral(data_batch, n=steps)
        outputs = [
            [{"text": state_t} for state_t in instance_txt]
            for instance_txt in completions_txt
        ]

        experiment.add_instances(
            inputs=data_batch,
            outputs=outputs,
        )


def run_quest(
    experiment,
    model,
    steps,
    data_batches,
    reward_model_batch_size=64,
    reward_device=1,
    reward_device_count=1,
    remote=False,
):

    reward_type = experiment.meta["reward_type"]

    if remote:
        registry = RegistryClient(
            store=FileSystemKVStore("/gscratch/ark/graf/registry"),
            max_history=3600,
            cache_ttl=60,
            service_type="model_path",
        )

        reward = RemoteReward(
            registry=registry,
            model_path=experiment.meta["reward_model_path"],
            reward_type=reward_type,
            # batch_size=reward_model_batch_size,
        )

    else:
        if reward_type == "contextual":
            reward = ContextualRewardModel(
                model_path=experiment.meta["reward_model_path"],
                # batch_size=reward_model_batch_size,
                device=reward_device,
                device_count=reward_device_count,
            )
        elif reward_type == "value":
            reward = ValueHead(
                model_path=experiment.meta["reward_model_path"],
                # batch_size=reward_model_batch_size,
                device=reward_device,
                device_count=reward_device_count,
            )  # sentiment model.
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

    # Process each batch
    for data_batch in data_batches:
        context = [model.get_prompt(**data) for data in data_batch]
        reward.set_context(context)

        chain = Quest(
            input_data=data_batch,
            proposal=RLHFSuffixProposal(
                model=model,
                reward=reward,
            ),
            beta=experiment.meta["beta"],
        )

        chain_outputs = chain.run(
            steps=steps,
            use_tqdm=True,
        )

        outputs = process_batch_outputs(chain_outputs, len(data_batch))
        experiment.add_instances(
            inputs=data_batch,
            outputs=outputs,
        )

    # Calculate and add reward scores
    scores = calculate_reward_scores(experiment)
    experiment.add_eval(reward.get_name(), scores)

    return experiment


def run_quest_bootstrap(
    experiment,
    model,
    steps,
    reward_device=1,
    reward_device_count=1,
    remote=False,
):

    reward_type = experiment.meta["reward_type"]

    if remote:
        registry = RegistryClient(
            store=FileSystemKVStore("/gscratch/ark/graf/registry"),
            max_history=3600,
            cache_ttl=60,
            service_type="model_path",
        )

        reward = RemoteReward(
            registry=registry,
            model_path=experiment.meta["reward_model_path"],
            reward_type=reward_type,
            # batch_size=reward_model_batch_size,
        )

    else:
        if reward_type == "contextual":
            reward = ContextualRewardModel(
                model_path=experiment.meta["reward_model_path"],
                # batch_size=reward_model_batch_size,
                device=reward_device,
                device_count=reward_device_count,
            )
        elif reward_type == "value":
            reward = ValueHead(
                model_path=experiment.meta["reward_model_path"],
                # batch_size=reward_model_batch_size,
                device=reward_device,
                device_count=reward_device_count,
            )  # sentiment model.
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

    for data_batch in chunked(
        experiment.meta["bootstrap"], experiment.get("batch_size")
    ):
        context = [data["input"]["prompt"] for data in data_batch]

        reward.set_context(context)

        chain = Quest(
            input_data=[data["input"] for data in data_batch],
            proposal=RLHFSuffixProposal(
                model=model,
                reward=reward,
            ),
            beta=experiment.meta["beta"],
        )

        chain_outputs = chain.run(
            steps=steps,
            use_tqdm=True,
            warm_start=[
                {"completion": data["completion"], "reward": data["reward"]}
                for data in data_batch
            ],
        )

        outputs = process_batch_outputs(chain_outputs, len(data_batch))

        experiment.add_instances(
            inputs=data_batch,
            outputs=outputs,
        )

    # Calculate and add reward scores
    scores = calculate_reward_scores(experiment)
    experiment.add_eval(reward.get_name(), scores)

    return experiment


#  Create model
def run_experiment(
    experiment,
    gpu_memory_utilization=0.95,
    device_count=1,
    reward_model_batch_size=64,
    reward_device=1,
    reward_device_count=1,
    remote=False,
):

    # Create model
    model = create_vllm_model(
        model_path=experiment.meta["model_path"],
        temperature=experiment.meta["temperature"],
        max_new_tokens=experiment.meta["max_new_tokens"],
        max_prompt_length=experiment.meta["max_prompt_length"],
        device_count=device_count,
        gpu_memory_utilization=gpu_memory_utilization,
        stop_tokens=experiment.meta["stop_tokens"],
        remote=remote,
    )

    completed = len(experiment.instances())

    # Get batched data with start index
    data_batches = get_batched_data(
        model=model,
        dataset_path=experiment.meta["dataset"],
        split=experiment.meta["split"],
        n=experiment.meta["n"],
        batch_size=experiment.meta.get("batch_size", 64),
        prompt_template=experiment.meta["prompt_template"],
        start_index=experiment.meta.get("i", 0),
        num_chains=experiment.meta.get("num_chains", 1),
        completed=completed,
        format=experiment.meta.get("format", "chat"),
        use_few_shot=experiment.meta.get("use_few_shot", False),
    )

    if experiment.meta["variant"] == "ancestral":
        run_ancestral(
            experiment=experiment,
            model=model,
            steps=experiment.meta["steps"],
            data_batches=data_batches,
        )
    elif experiment.meta["variant"] == "quest-rlhf":
        run_quest(
            experiment=experiment,
            model=model,
            steps=experiment.meta["steps"],
            data_batches=data_batches,
            reward_model_batch_size=reward_model_batch_size,
            reward_device=reward_device,
            reward_device_count=reward_device_count,
            remote=True,  # remote,
        )


#  Create model
def run_experiment_remote(
    experiment,
):

    # Create model
    model = create_vllm_model(
        model_path=experiment.meta["model_path"],
        temperature=experiment.meta["temperature"],
        max_new_tokens=experiment.meta["max_new_tokens"],
        max_prompt_length=experiment.meta["max_prompt_length"],
        device_count=1,
        gpu_memory_utilization=0.95,
        stop_tokens=experiment.meta["stop_tokens"],
        remote=True,
    )

    completed = len(experiment.instances())

    # Get batched data with start index
    if "bootstrap" in experiment.meta:

        if "bootstrap" in experiment.meta:
            run_quest_bootstrap(
                experiment=experiment,
                model=model,
                steps=experiment.meta["steps"],
                reward_device=0,
                reward_device_count=1,
                remote=True,
            )

    else:

        data_batches = get_batched_data(
            model=model,
            dataset_path=experiment.meta["dataset"],
            split=experiment.meta["split"],
            n=experiment.meta["n"],
            batch_size=experiment.meta.get("batch_size", 64),
            prompt_template=experiment.meta["prompt_template"],
            start_index=experiment.meta.get("i", 0),
            num_chains=experiment.meta.get("num_chains", 1),
            completed=completed,
            format=experiment.meta.get("format", "chat"),
            use_few_shot=experiment.meta.get("use_few_shot", False),
        )

        if experiment.meta["variant"] == "ancestral":
            run_ancestral(
                experiment=experiment,
                model=model,
                steps=experiment.meta["steps"],
                data_batches=data_batches,
            )
        elif experiment.meta["variant"] == "quest-rlhf":

            run_quest(
                experiment=experiment,
                model=model,
                steps=experiment.meta["steps"],
                data_batches=data_batches,
                # reward_model_batch_size=experiment.meta.get("batch_size", 64),
                reward_device=0,
                reward_device_count=1,
                remote=True,
            )
