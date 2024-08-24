import torch
import open_clip


def get_clip_model(model_name="ViT-L-14"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer, preprocess


async def encode_text(model, tokenizer, text):
    device = next(model.parameters()).device
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    return text_features.cpu().numpy()
