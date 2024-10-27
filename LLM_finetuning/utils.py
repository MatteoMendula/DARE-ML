from transformers import TrainerCallback
import os

class LossThresholdCallback(TrainerCallback):
    def __init__(self, threshold, output_dir, model, tokenizer):
        self.threshold = threshold
        self.output_dir = output_dir
        self.model = model
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, **kwargs):
        # Access evaluation metrics
        metrics = kwargs.get("metrics", {})
        eval_loss = metrics.get("eval_loss")
        
        # Check if the evaluation loss is below the threshold
        if eval_loss and eval_loss < self.threshold:
            print(f"Stopping training early as eval_loss {eval_loss} < threshold {self.threshold}")
            control.should_training_stop = True  # Signal trainer to stop

            # Save model checkpoint manually
            save_path = os.path.join(self.output_dir, f"checkpoint-early-stop")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"Checkpoint saved at {save_path} due to early stopping.")

        else:
            print(f"Eval loss {eval_loss} is not below threshold {self.threshold}. Continuing training.")


def prompt_instruction_format(sample):
  return f"""### Instruction:
    Use the Task below and the Input given to write the Response:

    ### Task:
    Summarize the Input

    ### Input:
    {sample['dialogue']}

    ### Response:
    {sample['summary']}
    """ 