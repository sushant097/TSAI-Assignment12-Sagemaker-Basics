import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([T.ToPILImage(),
                        T.Resize((224, 224)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# load model
def model_fn(model_dir):
    model = torch.jit.load(f"{model_dir}/model.scripted.pt")

    model.to(device).eval()
    
    return model

# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    data = json.loads(request_body)["inputs"]
    data = transform(np.array(data).astype(np.uint8)).unsqueeze(0).to(device)
    return data


# inference
def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
        prediction = F.softmax(prediction, dim=1)
    
    confidences, cat_ids = torch.topk(prediction, 5)
    outputs = {
        model.idx_to_class[idx.item()]: c.item() for c, idx in zip(confidences[0], cat_ids[0])
    }
    return outputs


# postprocess
def output_fn(outputs, content_type):
    assert content_type == "application/json"
    # Print top categories per image
    return outputs
