from PIL import Image
from image_vectors import ImageEmbedder


def test_image_embedding_extraction():
    image = Image.open("./tests/assets/example-image.jpg")
    image = image.convert("RGB")
    embedder = ImageEmbedder(model="vit_base_patch16_clip_224.openai")
    embedding = embedder.embed(image)
    assert embedding.shape == (1, 768), "Embedding shape is not correct"
