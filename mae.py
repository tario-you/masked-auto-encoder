import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tqdm import trange


# -------------------------------------------
# CIFAR-10 Data Loading Utilities
# -------------------------------------------


def load_cifar_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']  # shape: (10000, 3072)
    labels = batch[b'labels']  # list of length 10000
    # Reshape data to (N, 3, 32, 32) and convert to float in [0,1]
    data = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return data, labels


def load_cifar10(data_folder):
    # Load training batches
    train_data = []
    train_labels = []
    for i in range(1, 6):
        file_path = os.path.join(data_folder, f"data_batch_{i}")
        data, labels = load_cifar_batch(file_path)
        train_data.append(data)
        train_labels += labels
    # shape: (50000, 3, 32, 32)
    train_data = np.concatenate(train_data, axis=0)

    # Load test batch
    test_file = os.path.join(data_folder, "test_batch")
    test_data, test_labels = load_cifar_batch(test_file)

    # (Optionally, you can load batches.meta to get class names)
    return (train_data, train_labels), (test_data, test_labels)

# Custom Dataset for CIFAR-10


class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        data: numpy array of shape (N, 3, 32, 32)
        labels: list or array of labels
        transform: optional transform function
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]  # (3, 32, 32)
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        # Convert numpy array to torch tensor if not already
        if not torch.is_tensor(img):
            img = torch.tensor(img, dtype=torch.float)
        return img, label

    def __len__(self):
        return len(self.data)

# -------------------------------------------
# MAE Model (adapted for CIFAR-10, 32x32 images)
# -------------------------------------------
# Patch Embedding: splits image into patches


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # Use a conv layer to extract patches and project them
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

# Transformer block used in both encoder and decoder


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

# MAE Encoder: processes only visible (unmasked) patches


class MAEEncoder(nn.Module):
    def __init__(self, embed_dim=128, depth=6, num_heads=4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

# MAE Decoder: lightweight transformer to reconstruct masked patches


class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=128, decoder_dim=128, depth=4, num_heads=4,
                 patch_size=4, img_size=32, in_chans=3):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(decoder_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(
            decoder_dim, patch_size * patch_size * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.mask_token)
        nn.init.xavier_uniform_(self.pos_embed)

    def forward(self, x, ids_restore):
        B, N_vis, _ = x.shape
        x = self.decoder_embed(x)  # (B, N_vis, decoder_dim)
        N = self.num_patches
        N_mask = N - N_vis
        mask_tokens = self.mask_token.expand(B, N_mask, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, N, decoder_dim)
        # Restore the original order
        x_full = torch.gather(
            x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1)))
        x_full = x_full + self.pos_embed
        for block in self.blocks:
            x_full = block(x_full)
        x_full = self.norm(x_full)
        x_rec = self.decoder_pred(x_full)
        return x_rec


# Random masking helper function


def random_masking(x, mask_ratio):
    """
    x: (B, N, D) tokens
    Returns:
      x_masked: visible tokens (B, N_vis, D)
      mask: binary mask (B, N) with 0 for kept tokens and 1 for masked tokens
      ids_restore: indices to restore original order (B, N)
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(B, N, device=x.device)
    mask.scatter_(1, ids_keep, 0)
    return x_masked, mask, ids_restore

# Helper: Patchify function (for creating reconstruction targets)


def patchify(imgs, patch_size=4):
    """
    imgs: (B, C, H, W)
    Returns:
      patches: (B, num_patches, patch_size*patch_size*C)
    """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size"
    h = H // patch_size
    w = W // patch_size
    patches = imgs.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, h * w, -1)
    return patches

# Full MAE model combining encoder and decoder


class MaskedAutoencoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128,
                 encoder_depth=6, encoder_num_heads=4,
                 decoder_dim=128, decoder_depth=4, decoder_num_heads=4,
                 mask_ratio=0.75):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.mask_ratio = mask_ratio
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim))
        nn.init.xavier_uniform_(self.pos_embed)
        self.encoder = MAEEncoder(embed_dim, encoder_depth, encoder_num_heads)
        self.decoder = MAEDecoder(embed_dim, decoder_dim, decoder_depth, decoder_num_heads,
                                  patch_size, img_size, in_chans)

    def forward(self, imgs):
        x = self.patch_embed(imgs)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        x_masked, mask, ids_restore = random_masking(x, self.mask_ratio)
        latent = self.encoder(x_masked)
        x_rec = self.decoder(latent, ids_restore)
        return x_rec, mask

# Loss function: Compute MSE on masked patches only


def mae_loss(x_rec, imgs, mask, patch_size=4):
    target = patchify(imgs, patch_size)  # (B, num_patches, patch_dim)
    loss = (x_rec - target) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss

# -------------------------------------------
# Training Loop
# -------------------------------------------


def train_mae(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(enumerate(dataloader),
                        total=len(dataloader), desc="Training")

    for batch_idx, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        optimizer.zero_grad()
        x_rec, mask = model(imgs)
        loss = mae_loss(x_rec, imgs, mask, patch_size=4)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # if (batch_idx + 1) % 50 == 0:
        #     print(
        #         f"Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")

        # Update tqdm bar
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        progress_bar.update(1)

    avg_loss = running_loss / len(dataloader)
    return avg_loss


# Save checkpoint function
def save_checkpoint(model, optimizer, epoch, filepath="mae_checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch}")


# Load checkpoint function
def load_checkpoint(model, optimizer, filepath="mae_checkpoint.pth"):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found, starting from scratch.")
        return 0


# -------------------------------------------
# Main Script: Load Data, Create Model, and Train
# -------------------------------------------
if __name__ == "__main__":
    # Set device
    device = torch.device("mps")

    # Path to the CIFAR-10 folder (change this if needed)
    data_folder = "cifar-10-batches-py"

    # Load CIFAR-10 data
    (train_data, train_labels), (test_data,
                                 test_labels) = load_cifar10(data_folder)
    print("Loaded CIFAR-10 training data:", train_data.shape)

    # Create training dataset and dataloader
    train_dataset = CIFAR10Dataset(train_data, train_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4)

    # Instantiate the MAE model (for CIFAR-10, using 32x32 images)
    model = MaskedAutoencoder(
        img_size=32,
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        encoder_depth=6,
        encoder_num_heads=4,
        decoder_dim=128,
        decoder_depth=4,
        decoder_num_heads=4,
        mask_ratio=0.75
    ).to(device)

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.05)

    # Training settings
    num_epochs = 300

    # checks if checkpoint exists
    start_epoch = load_checkpoint(
        model, optimizer, filepath="mae_checkpoint.pth")

    # Training loop
    for epoch in trange(start_epoch, num_epochs, desc="Epochs"):
        avg_loss = train_mae(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        # Save model checkpoint
        save_checkpoint(model, optimizer, epoch + 1,
                        filepath="mae_checkpoint.pth")

    # (Optionally, add validation/testing code here)
