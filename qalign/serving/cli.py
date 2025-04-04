from qflow.serving.cluster import Cluster
from qflow.serving.clustermonitor import ClusterStateMonitor
from qflow.serving.slurm_utils import SlurmScriptGenerator
import time
import pprint
from termcolor import colored
import asyncio
from literegistry import RegistryClient, FileSystemKVStore

# RegistryClient


def check_registry(verbose=False):
    r = RegistryClient(FileSystemKVStore("/gscratch/ark/graf/registry"))

    pp = pprint.PrettyPrinter(indent=1, compact=True)

    for k, v in asyncio.run(r.models()).items():
        print(f"{colored(k, 'red')}")
        for item in v:
            print(colored("--" * 20, "blue"))
            for key, value in item.items():

                if key == "request_stats":
                    if verbose:
                        print(f"\t{colored(key, 'green')}:{value}")
                    else:
                        if "last_15_minutes_latency" in value:
                            nvalue = value["last_15_minutes"]
                            print(f"\t{colored(key, 'green')}:{colored(nvalue,'red')}")
                        else:
                            print(f"\t{colored(key, 'green')}:NO METRICS YET.")
                else:
                    print(f"\t{colored(key, 'green')}:{value}")

    # pp.pprint(r.get("allenai/Llama-3.1-Tulu-3-8B-SFT"))


def check_summary():
    r = RegistryClient(FileSystemKVStore("/gscratch/ark/graf/registry"))

    for k, v in asyncio.run(r.models()).items():
        print(f"{colored(k, 'red')} :{colored(len(v),'green')}")


def terminate_cluster():
    cluster = Cluster()
    cluster.terminate()


#
def launch_cluster(account="cse"):

    cluster = Cluster(account=account)  # Initialize with your config paths
    # cluster.terminate()
    # import pdb

    # pdb.set_trace()
    # Define multiple configurations
    configurations = [
        # {"device_name": "l40s-4", "script_spec": "tulusft", "target_instances": 6},
        # {"device_name": "a40-4", "script_spec": "tulusft", "target_instances":6},
        # {"device_name": "l40-4", "script_spec": "tulusft", "target_instances": 6},
        # {"device_name": "l40-4", "script_spec": "tuludpo", "target_instances": 2},
        # {"device_name": "l40s-4", "script_spec": "tuludpo", "target_instances": 2},
        # {"device_name": "l40-4", "script_spec": "tulurm", "target_instances": 6},
        # {"device_name": "l40s-4", "script_spec": "tulurm", "target_instances": 6},
        # {"device_name": "a40-4", "script_spec": "tulurm", "target_instances": 6},
        # {"device_name": "l40-4", "script_spec": "llama8b", "target_instances": 8},
        # {"device_name": "l40s-4", "script_spec": "llama8b", "target_instances": 16},
        # {"device_name": "a40-4", "script_spec": "llama8b", "target_instances": 16},
        {"device_name": "a40-4", "script_spec": "gsmrm", "target_instances": 16},
        {"device_name": "l40-4", "script_spec": "gsmrm", "target_instances": 8},
        {"device_name": "l40s-4", "script_spec": "gsmrm", "target_instances": 16},
        # {"device_name": "a40-1", "script_spec": "xqe", "target_instances": 16},
        # {"device_name": "l40-1", "script_spec": "xqe", "target_instances": 16},
        # {"device_name": "l40s-1", "script_spec": "xqe", "target_instances": 16},
        # {"device_name": "a40-4", "script_spec": "euro", "target_instances": 4},
        # {"device_name": "l40-4", "script_spec": "euro", "target_instances": 4},
        # {"device_name": "l40s-4", "script_spec": "euro", "target_instances": 4},
        # {"device_name": "a40-4", "script_spec": "tulusft", "target_instances": 2},
        # {"device_name": "a40-4", "script_spec": "llama8b", "target_instances": 1},
        # {"device_name": "a40-4", "script_spec": "tulusft", "target_instances": 1},
        # {"device_name": "l40-2", "script_spec": "llama8b", "target_instances": 1},
        # {"device_name": "l40-2", "script_spec": "llama8b", "target_instances": 4},
        # {"device_name": "l40s-2", "script_spec": "a40-4", "target_instances": 2},
        # {"device_name": "a100", "script_spec": "vllm", "target_instances": 1},
    ]

    # Create and start the monitor
    monitor = ClusterStateMonitor(
        cluster, configs=configurations, check_interval=60  # Check every minute
    )

    try:
        print("Starting cluster state monitor...")
        monitor.start()

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop()
        monitor.join()

        # Optionally terminate all jobs when stopping
        print("Terminating all jobs...")
        cluster.terminate()

        print("Monitor stopped successfully")


def ark_nodes(account="ark"):

    cluster = Cluster(account=account)  # Initialize with your config paths
    # cluster.terminate()
    # import pdb

    # pdb.set_trace()
    # Define multiple configurations
    configurations = [
        # {"device_name": "l40s-1", "script_spec": "tulusft", "target_instances": 2},
        {"device_name": "l40s-1", "script_spec": "tuludpo", "target_instances": 2},
        # {"device_name": "l40-4", "script_spec": "tulusft", "target_instances": 2},
        # {"device_name": "l40-2", "script_spec": "tuludpo", "target_instances": 2},
        {"device_name": "l40-2", "script_spec": "tulusft", "target_instances": 1},
        # {"device_name": "l40s-2", "script_spec": "tulurm", "target_instances": 1},
        {"device_name": "a40-2", "script_spec": "tulurm", "target_instances": 1},
        {"device_name": "l40-4", "script_spec": "llama3b", "target_instances": 1},
        # {"device_name": "a40-2", "script_spec": "tulusft", "target_instances": 1},
        # {"device_name": "a40-4", "script_spec": "llama8b", "target_instances": 1},
        # {"device_name": "a40-4", "script_spec": "tulusft", "target_instances": 1},
        # {"device_name": "l40-2", "script_spec": "llama8b", "target_instances": 1},
        # {"device_name": "l40-2", "script_spec": "llama8b", "target_instances": 4},
        # {"device_name": "l40s-2", "script_spec": "a40-4", "target_instances": 2},
        # {"device_name": "a100", "script_spec": "vllm", "target_instances": 1},
    ]

    for cfg in configurations:
        # print(cfg)
        # job_id = cluster.invoke(cfg.device_name, cfg.script_spec)
        # print(job_id)
        # cluster.
        print("" * 20)

        device = cluster.devices_specs[cfg["device_name"]]
        script_config = cluster.script_specs[cfg["script_spec"]]

        script = SlurmScriptGenerator(
            device_spec=device,
            script_config=script_config,
            account=account,
        )._generate_command()

        print(script)


def main(mode: str = "registry", account="cse"):
    if mode == "cluster":
        launch_cluster(account)
    elif mode == "registry":
        check_registry()
    elif mode == "show":
        ark_nodes(account)
    elif mode == "summary":
        check_summary()
    elif mode == "terminate":
        terminate_cluster()
    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
