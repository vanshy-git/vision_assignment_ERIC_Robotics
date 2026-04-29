import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from dataset import RoboticsDataset
from model import SimCLRModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the model
    model = SimCLRModel().to(device)
    model.load_state_dict(torch.load('outputs/trained_brain.pth'))
    model.eval() # Set to evaluation mode

    # 2. Get features for all images
    dataset = RoboticsDataset(image_dir='data/rgb/')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    features = []
    print("Extracting features...")
    with torch.no_grad():
        for v1, _ in loader:
            v1 = v1.to(device)
            # We take the raw backbone features, not the projection head
            feat, _ = model(v1)
            features.append(feat.cpu().numpy().flatten())

    features = np.array(features)

    # 3. Reduce dimensions
    print("Running PCA and t-SNE...")
    # PCA first to clean the noise
    pca_feats = PCA(n_components=50).fit_transform(features)
    # t-SNE to make it look good in 2D
    tsne_feats = TSNE(n_components=2, perplexity=30, init='pca').fit_transform(pca_feats)

    # 4. Plot the results
    plt.figure(figsize=(10, 7))
    # We color the dots by their index (time) to see the camera trajectory
    scatter = plt.scatter(tsne_feats[:, 0], tsne_feats[:, 1], c=range(len(tsne_feats)), cmap='viridis')
    plt.colorbar(scatter, label='Frame Sequence (Time)')
    plt.title('t-SNE Visualization of Learned Representations')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    plt.savefig('outputs/tsne_plot.png')
    print("Success! Plot saved as outputs/tsne_plot.png")
    plt.show()

if __name__ == '__main__':
    main()