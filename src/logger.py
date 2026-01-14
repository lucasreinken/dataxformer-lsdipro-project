"""
Precision Recall, F-Score
Top1, Top5, TopAll Acc for found solutions compared to the real one 
Statistics over NCols, NRows, Numel in Found and Relevant Tables 
Single vs Multi-Column 
Execution Time (Overall, Funcs?, Steps?)

(Hyperparameter Tuning without Multi-Hop?)
Erst Tau 
Dann Examples 5 Examples vs 25% vs 50% vs 75%  ()
Top-k (50 given? No info in third paper 6.3.2), make sure that it is not the same? Threshhold against Topk 
EM vs. Majority  
Dann Multi-Hop, (An-Aus, All iters, one, two, End?, Path-Len 1, 2, (3), Max-Tables (10, 25, 50, Inf))


Sollten wir non Non-Functional Dependencies nutzen? 
1:N 

LLM, wenn dann wie? 
Translations 
Web Forms 
Non-Functional Evaluation als Alternative zu k-nearest or """


import datetime
import logging
import os
import sys
import tempfile
import wandb
import subprocess
import platform 
from pathlib import Path
from time import perf_counter
from functools import wraps
from dataclasses import asdict
from pprint import pformat 

def get_rank(): 
    """Placeholder, if we would manage to distribute our framework, we would need to get a rank but for instance torch variants might not make sense"""
    return 0 

def get_git_commit_hash(short=True):

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

        suffix = " (dirty ⚠️)" if is_dirty else ""
        return f"{hash_output}{suffix}"

    except Exception:
        return "unknown"

###Brauche neue device func, da troch importieren zu viel RAM frist 

def login() -> None:
    """
    Helper func to log yourself in automatically.

    Args:
        None

    Returns:
        None
    """
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(api_key)

def get_environment_info() -> str:

    """Fehlt: Pandas Version, Numpy Version"""

    python_version = sys.version.split()[0]
    os_info = f"{platform.system()} {platform.release()}"

    return f"Python {python_version} | OS {os_info}"

def configure(log_dir:Path|str = None): 
    logging.getLogger().handlers = list()

    if log_dir is None: 
        log_dir = os.getenv("Directory")
    if log_dir is None: 
        log_dir = os.path.join(
            tempfile.gettempdir(), 
            datetime.datetime.now().strftime("DataXFormer-%Y-%m-%d-%H-%M-%S"),
        )
    
    assert isinstance(log_dir, (str, Path))
    log_dir = os.path.expanduser(log_dir)
    os.makedirs(os.path.expanduser(log_dir), exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    rank = get_rank() 

    if rank == 0: 
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging configured. Saving logs to: {log_dir}")
    else: 
        file_handler = logging.FileHandler(os.path.join(log_dir, f"log_rank{rank}.txt"))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return log_dir 
    

class LoggingContext: 

    def __init__(self, configs: list, project_name: str, entity, log_dir:Path|str = None, seed:int = 2): 
        
        self.start_time = None 
        self.end_time = None 
        self.configs = configs
        self.seed = seed 
        self.project_name = project_name 
        self.entity = entity
        self.dir = log_dir
    
    def log(self, data:dict, step:int | None = None, commit:bool = True): 
        if self.run is None: 
            return 
        self.run.log(data, step = step, commit = commit)

    def __enter__(self): 
        login()

        user_time = datetime.datetime.now() 
        self.start_time = self.end_time = perf_counter() 

        log_dir = configure(log_dir = log_dir)

        commit_hash = get_git_commit_hash()
        environment = get_environment_info()

        logging.info(f"Start Time: {user_time}")
        logging.info(f"Global Seed: {self.seed}")
        logging.info(f"Log Directory: {log_dir}")
        logging.info(f"Commit Commit: {commit_hash}")
        logging.info(f"Encironment Info: {environment}")

        #other stuff like git_commit_hash etc. 

        rank = get_rank() 
        wandb_config = dict()
        for config in self.configs: 
            name = type(config).__name__
            wandb_config[name] = asdict(config)
            logging.info(f"{name}: {pformat(wandb_config[name])}")
        wandb["global_seed"] = self.seed

        if rank ==0: 
            run_id = os.path.basename(log_dir)
            self.run = wandb.init(
                project = self.project_name, 
                entity = self.entity, 
                config = wandb_config, 
                name = run_id, 
                id = run_id, 
                resume = "allow", 
            )
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback): 
        self.end_time = perf_counter() 
        duration = self.end_time - self.start_time 
        duration_str = str(datetime.timedelta(seconds = int(duration))) 

        if get_rank() == 0 and self.run is not None: 
            if exc_type: 
                self.run.finish(exit_code = 1)
                logging.warning("Wandb Run marked as FAILED due to exception.")
            else: 
                self.run.finish(exit_code = 0)
                logging.info("Wandb Run finished successfully.")
        
        if exc_type: 
            logging.error(f"Run CRASHED after {duration_str}")
            logging.exception("Exception details:")
            return False 
        
        else: 
            logging.info("Run finished successfully.")
            logging.info(f"Total Duration: {duration_str}")

def timeit(func): 
    @wraps(func)
    def wrapper(*args, **kwargs): 
        start_time = end_time = perf_counter() 
        try: 
            results = func(*args, **kwargs)
            return results
        finally: 
            end_time = perf_counter()
            duration = end_time - start_time 
            logging.info(f"Duration of func{func.__name__}: {duration}")
        
    return wrapper



if __name__ == "__main__": 
    pass