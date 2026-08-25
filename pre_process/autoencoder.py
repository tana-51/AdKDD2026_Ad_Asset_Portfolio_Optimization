import os
import torch
import clip
from PIL import Image
from sklearn.decomposition import PCA
import pickle
import numpy as np
import tqdm
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import islice


image_dir = Path(__file__).resolve().parents[1] / "images"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

file_path = Path(__file__).resolve().parents[1]
df_all = pd.read_csv(file_path / "list/train_data_list.txt", sep="\t", header=None, names=["product", "creative", "date", "impression", "click"])
unique_creatives = df_all["creative"].unique().tolist()


embeddings = []
filenames = []

for fname in tqdm(unique_creatives):
    path = os.path.join(image_dir, fname)
    img = preprocess(Image.open(path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        embeddings.append(feat.cpu().numpy().flatten())
        filenames.append(fname)

embeddings = np.array(embeddings)  # shape: (N_images, 512)

pca = PCA(n_components=10)
embeddings_10d = pca.fit_transform(embeddings)  # shape: (N_images, 10)

feature_dict = {fname: vec for fname, vec in zip(filenames, embeddings_10d)}

with open(file_path / "image_features.pkl", "wb") as f:
    pickle.dump(feature_dict, f)

