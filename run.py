import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from accelerate import Accelerator
import torch.distributed as dist
from model_shards import DistilBertShard1, DistilBertShard2

# Initialize accelerator
accelerator = Accelerator()
rank = accelerator.process_index
world_size = accelerator.num_processes

def main():
    # Only 2 processes supported
    assert world_size == 2, "This script is designed for 2 processes."
    # Load tokenizer (only on rank 0 for simplicity)
    if rank == 0:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        text = "Hello, Hugging Face Accelerate!"
        inputs = tokenizer(text, return_tensors="pt")
    else:
        inputs = None
    # Broadcast input to both processes
    inputs = accelerator.broadcast_object_list([inputs])[0]
    # Load full model (for splitting)
    full_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    if rank == 0:
        shard = DistilBertShard1(full_model.distilbert)
    else:
        shard = DistilBertShard2(full_model.distilbert)
    shard.eval()
    with torch.no_grad():
        if rank == 0:
            # Run first shard
            hidden = shard(inputs["input_ids"], attention_mask=inputs["attention_mask"])
            # Send hidden to rank 1
            dist.send(hidden.cpu(), dst=1)
            if accelerator.is_main_process:
                print("[Rank 0] Sent hidden states to Rank 1.")
        else:
            # Receive hidden from rank 0
            shape = (inputs["input_ids"].shape[0], inputs["input_ids"].shape[1], 768)
            hidden = torch.empty(shape)
            dist.recv(hidden, src=0)
            # Run second shard
            logits = shard(hidden, attention_mask=inputs["attention_mask"])
            print(f"[Rank 1] Logits: {logits}")

if __name__ == "__main__":
    main()
