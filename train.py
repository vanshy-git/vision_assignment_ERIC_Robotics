import torch
from torch.utils.data import DataLoader
from dataset import RoboticsDataset
from model import SimCLRModel

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Windows needs the full path or correct relative path
    dataset = RoboticsDataset(image_dir='data/rgb/')
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SimCLRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting Pretraining on Windows...")

    for epoch in range(10):
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = -torch.nn.functional.cosine_similarity(z1, z2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Round {epoch+1} complete.")

    torch.save(model.state_dict(), 'outputs/trained_brain.pth')
    print("Done! Saved to outputs folder.")