# from calflops import calculate_flops
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

class Task:
    def __init__(self, task_id: int, model_name: str, training_time: int = 100, memory_required: int = 32):
        self.id = task_id
        self.model_name = model_name
        self.training_time = training_time
        self.memory_required = memory_required
        self.assigned_gpu = None        

    def assign_gpu(self, gpu):
        """
        Assign a GPU to the task.
        """
        self.assigned_gpu = gpu
