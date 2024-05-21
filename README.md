# image-vectors

image-vectors makes extracting embedding vectors from images as simple as possible.

## Get Started

Install image-vectors from PyPI:
```shell
pip install image-vectors
```
and start extracting embedding vectors:
```python
from PIL import Image
from image_vectors import ImageEmbedder


embedder = ImageEmbedder(model="vit_base_patch32_clip_224.openai")

image = Image.open(<path_to_image>)
embedding = embedder.embed(image)
```

## Available Embedding Models

### CLIP Models

| Model | Embedding Dimensions |
| -------- | -------- |
| `vit_base_patch32_clip_224.openai`   | 768 |
| `vit_base_patch16_clip_224.openai`   | 768 |

### DINO Models

| Model | Embedding Dimensions |
| -------- | -------- |
| `vit_small_patch14_dinov2.lvd142m`   | 384 |


### SAM Models

| Model | Embedding Dimensions |
| -------- | -------- |
| `samvit_base_patch16.sa1b`   | 256 |
