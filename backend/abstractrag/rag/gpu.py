"""Shared helper for freeing a model's GPU memory.

Loading a torch model onto CUDA keeps it resident until something explicitly
drops it - Python's own garbage collector does not know CUDA memory exists.
Used by anything that shares the GPU with the LLM: see engine.py, which frees
the embedder and reranker right before calling the LLM (all three fit
one-at-a-time in 6 GB, not all three loaded at once).
"""


def release_cuda_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
