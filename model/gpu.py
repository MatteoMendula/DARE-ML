class GPU:
    def __init__(self, gpu_id: int, memory_size: float):
        self.id = gpu_id
        self.memory_size = memory_size
        self.is_available = True
