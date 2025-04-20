import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

VAL_DIR = './data/ILSVRC2012_img_val'  # 每個子資料夾為一個 synset 編號
CLASS_INDEX = './imagenet_class_index.json'

# 載入 synset, index 映射
with open(CLASS_INDEX, 'r') as f:
    class_idx = json.load(f)
synset_to_idx = {v[0]: int(k) for k, v in class_idx.items()}
idx_to_label = [v[1] for k, v in sorted(class_idx.items(), key=lambda x: int(x[0]))]

# 建立文字 prompt
prompts = [f"a photo of a {label}" for label in idx_to_label]

# 載入 CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 前處理
transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])

# 入 ImageFolder 格式的 val set
imagenet_val = datasets.ImageFolder(root=VAL_DIR, transform=transform)
synset_to_folder_idx = imagenet_val.class_to_idx
folder_idx_to_pytorch_idx = [synset_to_idx[syn] for syn, _ in sorted(synset_to_folder_idx.items(), key=lambda x: x[1])]

val_loader = DataLoader(imagenet_val, batch_size=32, num_workers=24, pin_memory=True)

# 處理文字嵌入
with torch.no_grad():
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 執行準確率評估
top1, top5, total = 0, 0, 0
for images, targets in tqdm(val_loader):
    images = images.to(device)
    targets = targets.to(device)
    true_indices = torch.tensor([folder_idx_to_pytorch_idx[t.item()] for t in targets], device=device)

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits_per_image = image_features @ text_features.T

    top5_preds = logits_per_image.topk(5, dim=-1).indices
    top1 += (top5_preds[:, 0] == true_indices).sum().item()
    top5 += sum([t in p for t, p in zip(true_indices, top5_preds)])
    total += targets.size(0)

print(f"Top-1 Accuracy: {top1 / total:.4f}")
print(f"Top-5 Accuracy: {top5 / total:.4f}")
