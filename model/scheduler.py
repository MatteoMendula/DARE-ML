from typing import List
from model.gpu import GPU
from model.task import Task
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class Scheduler:
    def __init__(self, gpus: List[GPU]):
        self.gpus = gpus

    def assign_task_to_gpu(self, task: Task) -> bool:
        # Try to find an available GPU with sufficient memory
        for gpu in self.gpus:
            if gpu.is_available and task.memory_required <= gpu.memory_size:
                task.assign_gpu(gpu)
                gpu.is_available = False  # Mark GPU as in use
                print(Fore.GREEN + f"Task {task.id} assigned to GPU {gpu.id} with {gpu.memory_size} GB memory.\n")
                return True
        # print(Fore.RED + f"No available GPU found for Task {task.id}.\n")  # we do not consider parallelization yet
        return False

    def release_gpu(self, gpu_id: int):
        # Release the GPU after task completion
        for gpu in self.gpus:
            if gpu.id == gpu_id:
                gpu.is_available = True
                # print(f"GPU {gpu_id} released.\n")
                break
