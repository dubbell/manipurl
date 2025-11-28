import torch
from transformers import CLIPProcessor, CLIPModel


model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).cuda()
processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)



def project_text(text_to_project):
    """Project text into latent embedding space (512)."""

    text_token = processor(
        text=text_to_project,
        return_tensors="pt",
        padding=True)
    
    text_token = { k : v.cuda() for k, v in text_token.items() }

    with torch.no_grad():
        text_embedding = model.get_text_features(**text_token)

    return text_embedding


def project_image(image_to_project):
    """Project image into latent embedding space (512)."""

    image_token = processor(
        images=image_to_project)
    
    image_token = { k : v.cuda() for k, v in image_token.items() }
    
    with torch.no_grad():
        image_embedding = model.get_image_features(**image_token)

    return image_embedding