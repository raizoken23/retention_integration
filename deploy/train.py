import torch
from contextlib import nullcontext
from torch.utils.data import DataLoader

import os
import argparse
import time
import gc
import wandb
import logging
import models.powercoder  # noqa: F401
from typing import Any, Callable
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model_config = {
    'vocab_size': 50304,
    'num_hidden_layers': 12,
    'num_attention_heads': 12,
    'num_key_value_heads': 2,
    'hidden_size': 768,
    'bias': True,
}


def parse_args():
    parser = argparse.ArgumentParser(prog='train_power', description='Take an off-the-shelf model from huggingface and train a power model')
    # dataset
    parser.add_argument('--dataset', type=str, default='karpathy/tiny_shakespeare', help='Dataset to use: any hf dataset')
    # logging config
    parser.add_argument('--log-cabin', action='store_true', default=False)
    # model config
    parser.add_argument('--model', type=str, default='./models/powercoder')
    parser.add_argument('--tokenizer', type=str, default='bigcode/starcoder2-3b')
    parser.add_argument('--chunk_size', type=int, default=None, help='Chunk size to use: None for full sequence')
    parser.add_argument('--switch_over_seq_len', type=int, default=None, help='Sequence length threshold for chunked form')
    # training config
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--block_size', type=int, default=16384)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--no-amp', action='store_true', default=True)
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    # logging
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    return parser.parse_args()
    

def make_logger(args) -> Callable[..., None]:
    if args.log_cabin:
        from log_cabin_v2 import Dispatcher as LogCabinDispatcher
        username = os.getenv("LOG_CABIN_USER", os.getenv("USER", None))
        logger = LogCabinDispatcher(
            local_path=os.path.expanduser("~/.logs"),
            server_url="http://log-cabin-v2/ingestion",
            username=username,
        )
        logger.init(run_name=args.run_name, info=vars(args))
        run_name = logger.run_name
        return lambda **kwargs: logger.log('train', kwargs)
    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=run_name)
        return lambda **kwargs: wandb.log(kwargs)

    py_logger = logging.getLogger("train_power")
    if not py_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        py_logger.addHandler(handler)
        py_logger.setLevel(logging.INFO)
        py_logger.propagate = False

    BLUE = "\033[34m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    def log_metrics(**kwargs: Any) -> None:
        parts = []
        for k, v in kwargs.items():
            v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            parts.append(f"{BLUE}{k}{RESET}: {GREEN}{v_str}{RESET}")
        if parts:
            py_logger.info(" ".join(parts))

    return log_metrics


def make_dataloader(dataset_name: str, tokenizer: AutoTokenizer, batch_size: int, block_size: int):
    ds = datasets.load_dataset(dataset_name, split='train', trust_remote_code=True)
    # Check if dataset has a "text" column
    if "text" not in ds.features:
        raise ValueError("Dataset must have a 'text' column")

    # a simple tokenization and packing
    def tokenize(example):
        ids = tokenizer.encode(example['text'])
        return {'tokens': ids}
    all_columns = ds.column_names
    tokenized = ds.map(tokenize, batched=False, remove_columns=all_columns)

    def pack(examples):
        ids = sum(examples['tokens'], [])
        total_len = (len(ids) // block_size) * block_size
        ids = ids[:total_len]
        ids = [ids[i:i+block_size] for i in range(0, total_len, block_size)]
        return {'tokens': ids}
    
    packed = tokenized.map(pack, batched=True)

    def collate(batch):
        t = torch.stack([torch.tensor(ex["tokens"]) for ex in batch], dim=0).contiguous()
        return {"input_ids": t, "labels": t}

    return DataLoader(packed, batch_size=batch_size, collate_fn=collate, num_workers=1, pin_memory=True)


def release_unused_cuda_memory():
    torch.cuda.synchronize()   # ensure all kernels done
    gc.collect()               # free Python-side refs
    torch.cuda.empty_cache()   # return unreferenced cached blocks to the driver


def main():
    GREEN = "\033[32m"
    BLUE  = "\033[34m"
    RESET = "\033[0m"
    args = parse_args()
    torch.manual_seed(args.seed)
    device = 'cuda'
    logger = make_logger(args)
    print(f"{BLUE}args: {GREEN}{args.__dict__}{RESET}")

    cfg = AutoConfig.from_pretrained(args.model, **model_config)
    print(f"{BLUE}cfg: {GREEN}{cfg}{RESET}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    dl = make_dataloader(args.dataset, tokenizer, args.batch_size, args.block_size)

    model = AutoModelForCausalLM.from_config(cfg)
    
    print(f"{BLUE}model: {GREEN}{model}{RESET}")
    print(f"{BLUE}total_model_params: {GREEN}{sum(p.numel() for p in model.parameters())}{RESET}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    if args.compile:
        _ts = time.time()
        print(f"{BLUE}compiling the model...{RESET}")
        model = torch.compile(model)
        print(f"{BLUE}time to compile: {GREEN}{time.time() - _ts}{RESET}")

    model.train()
    model.to(device)

    start_time = time.time()
    batch_sample = []
    
    data_iter = iter(dl)
    def get_next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            return next(data_iter)

    for step in range(args.steps):
        optimizer.zero_grad()
        _start_time = time.time()
        for acc_step in range(args.grad_accum):
            with torch.autocast(device_type=device, dtype=torch.bfloat16) if not args.no_amp else nullcontext():
                batch = {k: v.to(device, non_blocking=True) for k, v in get_next_batch().items()}
                batch_sample.append(tokenizer.decode(batch['input_ids'][0][:20].tolist(), skip_special_tokens=True))
                outputs = model(**batch, chunk_size=args.chunk_size, switch_over_seq_len=args.switch_over_seq_len)
                loss = outputs.loss
            loss.backward()
        if args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        release_unused_cuda_memory()
        logger(iter=step, train_loss=float(loss.item()), lr=float(scheduler.get_last_lr()[0]), duration=(time.time() - _start_time) / 3600, train_hours=(time.time() - start_time) / 3600, avg_iter_per_hour=(1 / (time.time() - _start_time) * 3600))
        print(f"batch_sample: {batch_sample}")
        batch_sample = []



if __name__ == '__main__':
    main()