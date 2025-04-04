from qflow.serving.cluster import Cluster
from expkit import DiskStorage
from expkit.setup import ExpSetup
from typing import List, Dict
import time
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentManager:
    def __init__(
        self,
        cluster: Cluster,
        exp_setup: ExpSetup,
        max_parallel: int = 4,
        base_dir: str = "remote-outputs/",
        check_interval: int = 60,  # seconds
    ):
        self.cluster = cluster
        self.setup = exp_setup
        self.max_parallel = max_parallel
        self.base_dir = base_dir
        self.check_interval = check_interval

        # Queue of experiments to run
        self.exp_queue = deque(self.setup.keys())

        print(f"Experiments to run: {self.exp_queue}")

        # Track running experiments: {job_id: exp_id}
        self.running_jobs: Dict[str, str] = {}

    def launch_experiment(self, exp_id: str) -> str:
        """Launch a single experiment and return the job ID"""
        args = {
            "BASE_DIR": self.base_dir,
            "EXPERIMENT_NAME": exp_id,
        }

        job_id = self.cluster.invoke("node", "resume", **args)

        if job_id:
            logger.info(f"Launched experiment {exp_id} with job ID {job_id}")
            logger.info(f"cat slurm_{job_id}.log")
            logger.info(f"Experiment: {self.setup[exp_id]}\n{'-'*40}")

            self.running_jobs[job_id] = exp_id

            time.sleep(60)
            return job_id
        return None

    def check_and_update_jobs(self) -> List[str]:
        """Check status of running jobs and return completed job IDs"""
        completed_jobs = []

        files = []
        for job_id in list(self.running_jobs.keys()):
            status = self.cluster.check_job_status(job_id)
            files.append(f"slurm_{job_id}.log slurm_{job_id}.err")

            if status in [
                "COMPLETED",
                "FAILED",
                # "CANCELLED",
                # "TIMEOUT",
                "UNKNOWN",
            ]:
                exp_id = self.running_jobs[job_id]
                logger.info(
                    f"Experiment {exp_id} (job {job_id}) finished with status: {status}"
                )
                completed_jobs.append(job_id)
                del self.running_jobs[job_id]

            elif (status in ["CANCELLED", "TIMEOUT"]) or not status:
                exp_id = self.running_jobs[job_id]
                logger.info(
                    f"Experiment {exp_id} (job {job_id}) was canceled with status: {status}"
                )

                del self.running_jobs[job_id]

                self.exp_queue.append(exp_id)

        print("cat " + " ".join(files))

        return completed_jobs

    def agressive_lunch(
        self,
    ):

        while self.exp_queue and len(self.running_jobs) < self.max_parallel:
            next_exp = self.exp_queue.popleft()
            exp_id = next_exp
            print("Launching experiment:", exp_id)

            args = {
                "BASE_DIR": self.base_dir,
                "EXPERIMENT_NAME": exp_id,
            }

            job_id = self.cluster.run("node", "resume", **args)

            if job_id:
                logger.info(f"Launched experiment {exp_id} with job ID {job_id}")
                logger.info(f"cat slurm_{job_id}.log")
                logger.info(
                    f"Experiment {exp_id} launched with job ID {self.setup[exp_id]}\n{'-'*40}"
                )

                self.running_jobs[job_id] = exp_id

    def run(self):
        """Main loop to maintain K experiments running"""
        try:
            while self.exp_queue or self.running_jobs:
                # Check current jobs
                completed = self.check_and_update_jobs()

                # Launch new jobs if needed
                while len(self.running_jobs) < self.max_parallel and self.exp_queue:
                    next_exp = self.exp_queue.popleft()
                    self.launch_experiment(next_exp)

                # Log status
                logger.info(
                    f"Running jobs: {len(self.running_jobs)}, "
                    f"Remaining experiments: {len(self.exp_queue)}"
                )

                # Wait before next check
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("Received interrupt, cleaning up...")
            self.cluster.terminate()
        except Exception as e:
            logger.error(f"Error in experiment manager: {e}")
            self.cluster.terminate()
            raise


def main(dir="remote-outputs/"):

    # Initialize cluster
    cluster = Cluster(
        configs={
            "resume": "qflow/serving/configs/resume.yaml",
        },
    )

    # Initialize experiment setup
    setup = ExpSetup(DiskStorage(dir, mode="rw")).filter(
        lambda x: len(x.instances()) < x.get("n")
    )

    # setup = setup.query({"variant": "ancestral"}).query(
    #    {"model_path": "allenai/Llama-3.1-Tulu-3-8B-SFT"}
    # )

    # Create and run experiment manager
    manager = ExperimentManager(
        cluster=cluster,
        exp_setup=setup,
        max_parallel=8,  # Set number of parallel experiments
        base_dir=dir,
    )

    manager.agressive_lunch()
    # manager.run()


if __name__ == "__main__":

    import fire

    fire.Fire(main)
