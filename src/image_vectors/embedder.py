import timm
import torch

import numpy as np
from PIL import Image
from typing import Iterable, Union


class ImageEmbedder:
    def __init__(self, model: str = "vit_base_patch16_clip_224.openai") -> None:
        self.model = timm.create_model(model, pretrained=True, num_classes=0)
        self.model = self.model.eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def embed(self, images: Union[Image.Image, Iterable[Image.Image]]) -> Iterable[np.ndarray]:
        if type(images) == list:
            image_tensors = [self.transforms(image) for image in images]
            image_tensors = torch.stack(image_tensors)
        else:
            image_tensors = self.transforms(images)
            image_tensors = image_tensors.unsqueeze(0)

        output = self.model.forward_features(image_tensors)
        output = self.model.forward_head(output, pre_logits=True)
        return output.cpu().detach().numpy()
