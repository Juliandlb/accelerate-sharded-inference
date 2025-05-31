# Minimal Sharded Inference with Hugging Face Accelerate (CPU, 2 processes)

This is a minimal proof-of-concept for sharded inference of a Hugging Face Transformer model using [Accelerate](https://github.com/huggingface/accelerate) on CPU, simulating two machines (processes) on a single machine.

## Features
- Uses PyTorch, Transformers, and Accelerate
- Manually splits DistilBERT into two shards (layers 0–2 and 3–5)
- Each process loads and runs inference on its shard
- Simulates input passing from shard 1 to shard 2 using `torch.distributed`
- CPU-only, no GPU required

## Files
- `requirements.txt` — dependencies
- `accelerate_config.yaml` — config for CPU, 2 processes
- `model_shards.py` — model split logic
- `run.py` — main script for sharded inference
- `README.md` — this file

## Quickstart

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Accelerate**

   The provided `accelerate_config.yaml` is ready for CPU, 2 processes. You can also run:

   ```bash
   accelerate config
   ```

3. **Run sharded inference**

   ```bash
   accelerate launch --config_file=accelerate_config.yaml --num_processes=2 run.py
   ```

   You should see output from both processes, with logits printed from Rank 1.

## How it works
- The model is split into two parts: `DistilBertShard1` (embeddings + first 3 layers) and `DistilBertShard2` (last 3 layers + classifier).
- Rank 0 runs the first shard, sends hidden states to Rank 1.
- Rank 1 receives hidden states, runs the second shard, and prints logits.

## Notes
- This is a minimal demo for educational purposes.
- For real distributed inference, see [Accelerate documentation](https://huggingface.co/docs/accelerate/index).
