from typing import *

## quest
from quest.reward.model import ValueHead
from quest.reward.remote import RemoteReward, RemoteReward
from quest import (
    RewardModel,
    ContextualRewardModel,
)

## expkit
from expkit.setup import ExpSetup
from expkit.storage import DiskStorage

## qalign
from qalign.utils.eval import *

## literegistry
from literegistry import RegistryClient, FileSystemKVStore


def main(
    base_dir="remote-outputs-llama/",
    reward_model_path="lastnumber",
    model_path="allenai/tulu-2-7b",
    batch_size=248,
    context=True,
    query_args={},
    device_count=8,
    value_head=True,
    remote=False,
    n=None,
):

    print("Query Args:", query_args)
    setup = ExpSetup(storage=DiskStorage(base_dir=base_dir, mode="rw"))
    # print("Exp:", setup.query(query_args))

    setup = setup.query(query_args).filter(lambda x: x.has_data())

    print("That match the query:\n", setup)

    if len(setup.experiments) == 0:
        raise FileNotFoundError("The experiment has no data!")

    if reward_model_path == "likelihood":

        ps_eval = LikelihoodEval(model_path=model_path)

    elif reward_model_path == "lastnumber":
        ps_eval = ExactLastNumberEval()

    elif reward_model_path == "lastmath":
        ps_eval = ExactMATHEval()

    elif reward_model_path == "lastoption":
        ps_eval = ExactQAEval()

    elif reward_model_path == "ifeval":
        ps_eval = IFEval()

    elif remote:

        if value_head:
            reward_type = "value"
        elif context:
            reward_type = "contextual"
        else:
            reward_type = "reward"

        registry = RegistryClient(
            store=FileSystemKVStore("/gscratch/ark/graf/registry"),
            max_history=3600,
            cache_ttl=60,
            service_type="model_path",
        )

        reward = RemoteReward(
            model_path=reward_model_path,
            registry=registry,
            reward_type=reward_type,
            # batch_size=batch_size,
            # max_parallel_requests=32,
            batch_size=32,
            max_parallel_requests=64,
        )
        ps_eval = RewardEval(
            reward=reward,
            n=n,
            chunk_size=256,
        )
    elif value_head:

        reward = ValueHead(
            model_path=reward_model_path,
            batch_size=batch_size,
            device_count=device_count,
        )

        ps_eval = RewardEval(reward=reward, n=n)

    else:
        if context:
            reward = ContextualRewardModel(
                model_path=reward_model_path,
                batch_size=batch_size,
                device_count=device_count,
            )

        else:
            reward = RewardModel(
                model_path=reward_model_path,
                batch_size=batch_size,
                device_count=device_count,
            )

        ps_eval = RewardEval(reward=reward, n=n)

    # setup = setup.filter(lambda x: not x.has_eval(ps_eval.eval_name))
    # setup = setup.filter(lambda x: x.get("n") == len(x.instances()))
    print("That haven't done the eval:", setup)

    def func(experiment):

        try:
            return (
                ps_eval(experiment)
                # if not experiment.has_eval(ps_eval.eval_name)
                # else experiment
            )
        except FileNotFoundError:
            return experiment

        except Exception as e:
            raise e
            # return experiment

    setup = setup.map(func)

    # new_setup.save()


if __name__ == "__main__":

    import fire

    fire.Fire(main)
