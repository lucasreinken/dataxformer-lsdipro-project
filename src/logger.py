import os
import sys
import platform
import random
import multiprocessing
import subprocess
import shutil
import wandb
import logging
import datetime
import tempfile
from importlib import metadata
from pathlib import Path
from time import perf_counter
from functools import wraps
from dataclasses import asdict
from pprint import pformat


def get_rank() -> int:
    """
    Returns the rank of the current process.

    Args:
        None

    Returns:
        int
    """

    for env_var in ["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK"]:
        if env_var in os.environ:
            return int(os.environ[env_var])

    try:
        identity = multiprocessing.current_process()._identity
        if identity:
            return identity[0]
    except (AttributeError, IndexError):
        pass

    return 0


def get_device() -> str:
    """
    Returns the current device.

    Args:
        None

    Returns:
        str
    """

    if shutil.which("nvidia-smi"):
        return "gpu"
    return "cpu"


def get_environment_info() -> str:
    """
    Returns the versions of the most important libraries.

    Args:
        None

    Returns:
        str
    """

    python_version = sys.version.split()[0]
    os_info = f"{platform.system()} {platform.release()}"

    libs = ["pandas", "numpy", "nltk", "streamlit", "orjson", "vertica-python"]
    versions = list()

    for lib in libs:
        try:
            v = metadata.version(lib)
            versions.append(f"{lib}=={v}")
        except metadata.PackageNotFoundError:
            versions.append(f"{lib} (not installed)")

    lib_str = " | ".join(versions)
    return f"Python {python_version} | OS {os_info} | {lib_str}"


def set_seed(seed: int = 2) -> None:
    """
    Sets the seed for all relevant libraries.

    Args:
        seed: int = 2

    Returns:
        None
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def get_git_commit_hash(short: bool = True) -> str:
    """
    Returns information about the currently used Git version.

    Args:
        short: bool = True

    Returns:
        str
    """

    try:
        cmd = ["git", "rev-parse", "HEAD"]
        if short:
            cmd.insert(2, "--short")

        hash_output = (
            subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            .strip()
            .decode("utf-8")
        )

        dirty_check = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        is_dirty = len(dirty_check) > 0

        suffix = " (dirty)" if is_dirty else ""
        return f"{hash_output}{suffix}"

    except Exception:
        return "unknown"


def login() -> None:
    """
    Helper func to log yourself in automatically.

    Args:
        None

    Returns:
        None
    """
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)


def configure(log_dir: Path | str | None = None) -> Path:
    """
    Configures the root logger.

    Args:
        log_dir: Path | str | None = None

    Returns:
        Path
    """
    logging.getLogger().handlers = list()

    if log_dir is None:
        log_dir = os.getenv("Directory")

    if log_dir is None:
        log_dir = os.path.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime("DataXFormer-%Y-%m-%d-%H-%M-%S"),
        )

    log_path = Path(log_dir).expanduser().resolve()
    log_path.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    rank = get_rank()

    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_path / "log.txt")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging configured. Saving logs to: {log_path}")
    else:
        file_handler = logging.FileHandler(log_path / f"log_rank{rank}.txt")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return log_path


class LoggingContext:
    """
    Logger for local filesystem logging and Weights & Biases integration.
    """

    def __init__(
        self,
        configs: list,
        project_name: str,
        entity: str | None,
        log_dir: Path | str | None = None,
    ) -> None:
        """
        Initializes the logging context.

        Args:
            configs: list
            project_name: str
            entity: str | None (W&B entity.)
            log_dir: Path | str | None
            seed: int = 2

        Returns:
            None
        """

        self.start_time = None
        self.end_time = None
        self.configs = configs
        self.seed = configs[-1].seed if configs else 2
        self.project_name = project_name
        self.entity = entity
        self.dir = log_dir
        self.run = None
        self.results_data = list()
        self.metrics_buffer = list()

    def info(self, msg: str) -> None:
        """
        Logs an info message locally.

        Args:
            msg: str

        Returns:
            None
        """

        logging.info(msg)

    def warning(self, msg: str) -> None:
        """
        Logs a warning message locally.

        Args:
            msg: str

        Returns:
            None
        """

        logging.warning(msg)

    def error(self, msg: str) -> None:
        """
        Logs an error message locally.

        Args:
            msg: str

        Returns:
            None
        """

        logging.error(msg)

    def log(
        self,
        data: dict,
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """
        Logs metrics to Weights & Biases if run is active.

        Args:
            data: dict (Dictionary of metrics to log.)
            step: int (Current step in the process.)
            commit: bool (Whether to finish the current log step.)

        Returns:
            None
        """

        if self.run is None:
            return
        self.run.log(data, step=step, commit=commit)

    def add_eval_result(self, name: str, metrics: dict) -> None:
        """
        Collects metrics over steps to log correctly in a distributed setting.

        Args:
            name: str
            metrics: dict

        Returns:
            None
        """

        row = [name] + [
            metrics.get(k, 0)
            for k in [
                "precision",
                "recall",
                "f1_score",
                "topk_acc",
                "topall_acc",
                "answered_rate",
            ]
        ]
        if "calc_time" in metrics:
            row.append(metrics["calc_time"])

        self.results_data.append(row)
        self.metrics_buffer.append(metrics)

        step = len(self.results_data)
        wandb_metrics = {f"live/{k}": v for k, v in metrics.items()}
        self.log(wandb_metrics, step=step)

    def __enter__(self):
        """
        Sets up the logging environment, local files, and W&B run.

        Args:
            None

        Returns:
            LoggingContext: (Initialized context instance.)
        """
        login()
        set_seed(self.seed)

        user_time = datetime.datetime.now()
        self.start_time = self.end_time = perf_counter()

        log_dir = configure(log_dir=self.dir)

        device = get_device()
        commit_hash = get_git_commit_hash()
        environment = get_environment_info()

        logging.info(f"Start Time: {user_time}")
        logging.info(f"Device: {device}")
        logging.info(f"Global Seed: {self.seed}")
        logging.info(f"Log Directory: {log_dir}")
        logging.info(f"Git Commit: {commit_hash}")
        logging.info(f"Environment Info: {environment}")

        rank = get_rank()
        wandb_config = dict()
        for config in self.configs:
            name = type(config).__name__
            wandb_config[name] = asdict(config)
            logging.info(f"{name}: {pformat(wandb_config[name])}")
        wandb_config["global_seed"] = self.seed

        if rank == 0:
            run_id = os.path.basename(log_dir)
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                config=wandb_config,
                name=run_id,
                id=run_id,
                resume="allow",
                settings=wandb.Settings(start_method="spawn"),
            )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Cleans up and shuts down the logging.

        Args:
            exc_type: Exception type if occurred.
            exc_value: Exception value.
            traceback: Traceback object.
        """

        self.end_time = perf_counter()
        duration = self.end_time - self.start_time
        duration_str = str(datetime.timedelta(seconds=int(duration)))
        rank = get_rank()

        if not exc_type and rank == 0 and self.run is not None:
            import numpy as np

            columns = [
                "Exercise",
                "Precision",
                "Recall",
                "F1_Score",
                "TopK_Acc",
                "TopAll_Acc",
                "Answered",
            ]

            if len(self.results_data) > 0 and len(self.results_data[0]) > 7:
                columns.append("Calc_Time")

            results_table = wandb.Table(columns=columns, data=self.results_data)
            self.log({"Results_Table": results_table})

            if self.metrics_buffer:
                keys = self.metrics_buffer[0].keys()
                summary = dict()

                for k in keys:
                    vals = [m[k] for m in self.metrics_buffer if m.get(k) is not None]
                    if vals:
                        summary[f"final_mean/{k}"] = np.mean(vals)

                self.log(summary)
                logging.info(f"Final Summary: {summary}")

        if rank == 0 and self.run is not None:
            if exc_type:
                self.run.finish(exit_code=1)
                logging.warning("Wandb Run marked as FAILED due to exception.")
            else:
                self.run.finish(exit_code=0)
                logging.info("Wandb Run finished successfully.")

        if exc_type:
            logging.error(f"Run CRASHED after {duration_str}")
            logging.exception("Exception details:")
            return False
        else:
            logging.info("Run finished successfully.")
            logging.info(f"Total Duration: {duration_str}")
            return True


def timeit(func):
    """
    Decorator to track the duration of a function.

    Args:
        func

    Returns:
        wrapper
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        try:
            results = func(*args, **kwargs)
            return results
        finally:
            end_time = perf_counter()
            duration = end_time - start_time
            logging.info(
                f"Duration of function '{func.__name__}': {duration:.4f} seconds"
            )

    return wrapper
