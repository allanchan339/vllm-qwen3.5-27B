# Versioning Issue
The vllm v0.19.0 is partial support for transformers 5. Due to some bugs, the vllm is still using transformers 4.49 

However, new model e.g. Qwen3.5-27B require new RoPE implementation, making transformers 5.3 or above is a must. 

You have to manually install transformers 5.3 or above, after installed vllm v0.19.0. e.g. 
```
uv pip install -U transformers
```

# 