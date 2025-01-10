# Power Attention

A PyTorch extension implementing symmetric power transformers - a variant of linear transformers that achieves transformer-level performance while scaling linearly with sequence length. This package provides efficient CUDA kernels that make it possible to process much longer sequences compared to standard quadratic attention.

For details on the approach, see our paper: [Symmetric Power Transformers](https://manifestai.com/articles/symmetric-power-transformers/)

## Installation

### From PyPI (Recommended)
```bash
pip install power-attention
```

### From Source
Requirements:
- Python 3.11+
- CUDA Toolkit 12.4
- GCC/G++ with C++17 support
- Linux (Windows/MacOS not supported)

```bash
git clone https://github.com/manifest-ai/power-attention.git
cd power-attention
pip install .
```

All other dependencies (PyTorch, Ninja build system, etc.) will be automatically installed through pip.

## Usage

The main entry point is the `power_full` function, which implements symmetric power attention. Here's a basic example:

```python
import torch
from power_attention import power_full

# Create input tensors
batch_size = 2
seq_len = 1024
num_heads = 8
head_dim = 64

Q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
K = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
V = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)

# Optional gating tensor (if using gated attention)
log_G = None  # or torch.randn(batch_size, seq_len, num_heads, dtype=torch.float32, device='cuda')

# Compute attention
output = power_full(
    Q=Q, K=K, V=V, 
    log_G=log_G,          # Optional gating tensor
    deg=4,                # Power attention degree (4 recommended for best performance)
    chunk_size=128,       # Size of chunks for processing long sequences
    deterministic=True,   # Whether to use deterministic algorithms
    normal_space=True     # Whether to use normal space (vs log space)
)
```

### Integration with Transformer Models

The package includes a drop-in replacement for standard attention in transformer models. See `training/model.py` for a complete example of using power attention in a GPT-style model:

```python
from power_attention import power_full

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... initialization code ...
        
    def forward(self, x):
        # ... projection code ...
        
        # Use power attention instead of standard attention
        y = power_full(
            q, k, v, 
            log_g,
            deg=self.degree,
            chunk_size=self.chunk_size
        )
        
        # ... output projection ...
        return y
```

## Features

- Efficient chunked algorithm for linear scaling with sequence length (O(t) cost vs O(t²) for standard attention)
- Support for gated attention and rotary embeddings
- CUDA kernels optimized for A100
- FP16 and BF16 support
- Replacement for standard attention in transformer models is possible for fine-tuning

## Development

### Setup

For development, first install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install development dependencies:

```bash
uv sync --dev --train --benchmark
```

This will install all dependencies including testing, training, and benchmarking tools.

To run tests:
```bash
make test
```

To run benchmarks:
```bash
make benchmark
```

For faster development iterations, you can use:
```bash
make fast  # Builds with optimized settings for development
```

### Training Example

After installing with training dependencies, you can run the training script:

```bash
# Single GPU training
python training/train.py \
  --batch_size=32 \
  --attention_kernel=power \
  --degree=4 \
  --chunk_size=128 \
  --out_dir=out/my_model

# Multi-GPU training with DDP (example with 4 GPUs)
torchrun --standalone --nproc_per_node=4 training/train.py \
  --batch_size=32 \
  --attention_kernel=power \
  --degree=4 \
  --chunk_size=128 \
  --out_dir=out/my_model
```

Key training parameters:
- `attention_kernel`: Use 'power' for symmetric power attention (default is 'sdpa' for standard attention)
- `degree`: Power attention degree (4 recommended)
- `chunk_size`: Size of chunks for processing long sequences
- `disable_gating`: Set to true to disable gating mechanism
- `log_space`: Whether to use log space computations

## Contributing

We welcome contributions! Here's how you can help:

### Getting Started

1. Fork the repository
2. Set up your development environment following the instructions above
3. Create a new branch for your feature/fix: `git checkout -b feature-name`

### Guidelines

- **Code Style**: Follow PEP 8 for Python code. For CUDA code, follow the existing style in the codebase
- **Documentation**: Add docstrings to new functions and update README if needed
- **Testing**: Add tests for new features and ensure all tests pass
- **Commits**: Write clear, concise commit messages
- **Performance**: For CUDA kernels, include benchmarks showing performance impact

### Pull Request Process

1. Update documentation for any new features
2. Add or update tests as needed
3. Ensure all tests pass: `make test`
4. Run benchmarks if performance-critical code was changed: `make benchmark`
5. Create a Pull Request with a clear description of changes
6. Wait for review and address any feedback

### Areas for Contribution

- Performance optimizations for different GPU architectures
- Documentation improvements
- Bug fixes
- Test coverage improvements

For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{buckman2024symmetric,
  title={Symmetric Power Transformers},
  author={Buckman, Jacob and Gelada, Carles and Zhang, Sean},
  publisher={Manifest AI},
  year={2024},
  month={8},
  url={https://manifestai.com/articles/symmetric-power-transformers/}
}
```

## License

MIT License
