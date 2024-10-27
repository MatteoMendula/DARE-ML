def estimate_training_time(self) -> float:
    """
    Estimate training time using `calculate_flops` from `calflops`.
    """
    try:
        # Load model and tokenizer based on model_name
        # model = AutoModelForCausalLM.from_pretrained(self.model_name)
        # tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Calculate FLOPs using calflops
        # flops, macs, params = calculate_flops(
        #     model=model,
        #     input_shape=(self.batch_size, self.max_seq_length),
        #     transformer_tokenizer=tokenizer
        # )
        flops, macs, params = 1,1,1 # TODO
        
        # Example calculation: converting FLOPs to an arbitrary training time
        print(f"Model: {self.model_name} FLOPs: {flops} TFLOPs, MACs: {macs} GMACs, Params: {params} B")
        return flops / (10**12)  # Convert FLOPS to seconds (adjust as needed)
    
    except Exception as e:
        print(f"Error estimating FLOPs for model {self.model_name}: {e}")
        return 0.0
