## qalign
from qalign.utils.experiment import run_experiment

## expkit
from expkit.storage import DiskStorage
from expkit import Exp


def main(
    experiment_name="",
    save_path: str = "llama3.2-outputs/",
    reward_model_batch_size: int = 128,
    gpu_memory_utilization: float = 0.95,
    reward_device=1,
    device_count: int = 1,
    remote=False,
):

    storage = DiskStorage(save_path, "rw")

    if storage.exists(experiment_name):
        experiment = Exp.load(storage=storage, name=experiment_name)

        run_experiment(
            experiment=experiment,
            gpu_memory_utilization=gpu_memory_utilization,
            device_count=device_count,
            reward_model_batch_size=reward_model_batch_size,
            reward_device=reward_device,
            remote=remote,
        )

    else:
        raise ValueError(f"Experiment {experiment_name} does not exist.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
