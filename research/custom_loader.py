import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomStegoDataset(Dataset):
    def __init__(self, images_folder, messages_folder, metadata_file, transform=None):
        self.images_folder = images_folder
        self.messages_folder = messages_folder
        self.metadata = pd.read_csv(metadata_file)
        
        # Default transformation for grayscale images with resizing
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to 224x224
                transforms.Grayscale(num_output_channels=3),  # Convert to RGB with 3 channels
                transforms.ToTensor(),           # Convert to tensor
            ])
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.images_folder, self.metadata.iloc[idx]['images'])
        image = Image.open(image_path).convert('RGB')  # Load as RGB first
        if self.transform:
            image = self.transform(image)  # Apply transformations
        
        # Load message
        message_path = os.path.join(self.messages_folder, self.metadata.iloc[idx]['message_files'])
        with open(message_path, 'r') as file:
            message = file.read()

        return image, message

# Function to return train/validation loaders
def get_loaders(images_folder, messages_folder, metadata_file, batch_size=8, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ensure images are resized to 224x224
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel RGB
        transforms.ToTensor(),           # Convert to PyTorch tensor
    ])

    dataset = CustomStegoDataset(images_folder, messages_folder, metadata_file, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
