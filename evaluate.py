import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataset import RoboticsDataset
from model import SimCLRModel

def get_labels(num_samples):
    # We split 500 frames into 3 simple zones: Start, Middle, End
    zone_size = num_samples // 3
    labels = [0]*zone_size + [1]*zone_size + [2]*(num_samples - 2*zone_size)
    return torch.LongTensor(labels)

def train_classifier(backbone, features, labels, title):
    # A tiny "head" to solve the 3-zone task
    classifier = nn.Linear(512, 3).to(features.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Train for a few rounds
    for epoch in range(50):
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Check accuracy
    _, predicted = torch.max(outputs, 1)
    acc = (predicted == labels).sum().item() / len(labels)
    print(f"Accuracy for {title}: {acc*100:.1f}%")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Pretrained Features
    model = SimCLRModel().to(device)
    model.load_state_dict(torch.load('outputs/trained_brain.pth'))
    model.eval()

    # 2. Prepare Data
    dataset = RoboticsDataset(image_dir='data/rgb/')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_feats = []
    with torch.no_grad():
        for v1, _ in loader:
            f, _ = model(v1.to(device))
            all_feats.append(f)
    
    features = torch.cat(all_feats)
    labels = get_labels(len(features)).to(device)

    # 3. Compare!
    print("--- Comparing Representations ---")
    
    # Test Pretrained
    train_classifier(model.backbone, features, labels, "Pretrained Model")
    
    # Test Random (Scratch)
    random_model = SimCLRModel().to(device) # New random weights
    random_feats = []
    with torch.no_grad():
        for v1, _ in loader:
            f, _ = random_model(v1.to(device))
            random_feats.append(f)
    
    train_classifier(random_model.backbone, torch.cat(random_feats), labels, "Model from Scratch")

if __name__ == '__main__':
    main()