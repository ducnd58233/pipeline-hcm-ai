import torch
import clip


def get_clip_model(model_name="ViT-B/32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(model_name, device=device)
    return model


def encode_text(model, text):
    device = next(model.parameters()).device
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features.cpu().numpy()
