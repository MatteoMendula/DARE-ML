from typing import List
from model.gpu import GPU
from model.task import Task
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class Scheduler:
    def __init__(self, gpus: List[GPU]):
        self.gpus = {}
        self.running_tasks = []
        for g in gpus:
            self.gpus[g.id] = g

    def assign_task_to_gpu(self, task: Task) -> bool:

        task_id = task.id.split("_retrain_")[0]
        if task_id in self.running_tasks:
            return False

        # Try to find an available GPU with sufficient memory
        for gpu_id in self.gpus.keys():
            gpu = self.gpus[gpu_id]
            if gpu.is_available and task.memory_required <= gpu.memory_size:
                task.assign_gpu(gpu)
                self.running_tasks.append(task_id)
                gpu.is_available = False  # Mark GPU as in use
                print(Fore.GREEN + f"Task {task.id} assigned to GPU {gpu_id} with {gpu.memory_size} GB memory.\n")
                return True
        # print(Fore.RED + f"No available GPU found for Task {task.id}.\n")  # we do not consider parallelization yet
        return False

    def release_gpu(self, gpu_id: int, task_id: str):
        task_id = task_id.split("_retrain_")[0]
        # Release the GPU after task completion
        self.gpus[gpu_id].is_available = True
        self.running_tasks.remove(task_id)
