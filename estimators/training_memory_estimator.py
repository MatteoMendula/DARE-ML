def estimate_memory_for_task(self) -> float:
    # Load the model to get the number of parameters
    model = AutoModel.from_pretrained(self.model_name)
    num_parameters = sum(p.numel() for p in model.parameters())
    
    # Assume FP32 (4 bytes per parameter)
    bytes_per_parameter = 4
    
    # Memory for model parameters
    memory_for_parameters = num_parameters * bytes_per_parameter
    
    # Estimate the size of activations (simplified)
    activations_size = self.batch_size * self.max_seq_length * (bytes_per_parameter * 128)  # Assuming 128 bytes per activation (this can vary)

    # Memory for gradients (same as parameters)
    gradients_size = num_parameters * bytes_per_parameter
    
    # Optimizer states (for Adam, we need 2x the size of parameters)
    optimizer_states_size = num_parameters * bytes_per_parameter * 2
    
    # Input tensor size
    input_size = self.batch_size * (self.max_seq_length * bytes_per_parameter)
    
    # Total memory required in bytes
    total_memory = (memory_for_parameters + activations_size + gradients_size + optimizer_states_size + input_size)
    
    # Convert bytes to GB
    return total_memory / (1024 ** 3)  # Convert to GB
