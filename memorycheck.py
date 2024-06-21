import torch

if torch.cuda.is_available():
    cuda_info = torch.cuda.get_device_properties(0)
    print(f"GPU Name: {cuda_info.name}")
    print(f"Total GPU Memory: {cuda_info.total_memory / 1024**2:.2f} MB")
    
    # Check GPU memory usage
    max_memory_allocated = torch.cuda.max_memory_allocated(0) / 1024**2  # Convert to MB
    max_memory_cached = torch.cuda.max_memory_cached(0) / 1024**2  # Convert to MB

    # Calculate free memory as total memory - used memory
    free_memory = cuda_info.total_memory / 1024**2 - max_memory_allocated
    print(f"Used GPU Memory: {max_memory_allocated:.2f} MB")
    print(f"Cached GPU Memory: {max_memory_cached:.2f} MB")
    print(f"Free GPU Memory: {free_memory:.2f} MB")
else:
    print("No GPU available.")
