import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# -------------------------------
# Model Components (adapted for CIFAR-10)
# -------------------------------


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


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
        x_full = torch.gather(
            x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1)))
        x_full = x_full + self.pos_embed
        for block in self.blocks:
            x_full = block(x_full)
        x_full = self.norm(x_full)
        x_rec = self.decoder_pred(x_full)
        return x_rec


def random_masking(x, mask_ratio):
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


def unpatchify(patches, patch_size=4, img_size=32, in_chans=3):
    B, N, patch_dim = patches.shape
    h = w = img_size // patch_size
    patches = patches.reshape(B, h, w, in_chans, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    imgs = patches.reshape(B, in_chans, img_size, img_size)
    return imgs


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
        x = self.patch_embed(imgs)
        x = x + self.pos_embed
        x_masked, mask, ids_restore = random_masking(x, self.mask_ratio)
        latent = self.encoder(x_masked)
        x_rec = self.decoder(latent, ids_restore)
        return x_rec, mask

# -------------------------------
# Inference Script
# -------------------------------


def main(input_image_path, output_image_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        # If checkpoint is a dict with a "model_state_dict" key, extract it.
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        # Remove a possible "module." prefix from keys.
        new_state_dict = {}
        for k, v in ckpt.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        print("Loaded pretrained model weights.")
    else:
        print("No checkpoint found. Using randomly initialized model.")

    model.eval()

    img = Image.open(input_image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        x_rec, mask = model(img_tensor)

    rec_img_tensor = unpatchify(x_rec, patch_size=4, img_size=32, in_chans=3)
    rec_img_tensor = torch.clamp(rec_img_tensor, 0, 1)

    rec_img = transforms.ToPILImage()(rec_img_tensor.squeeze(0).cpu())
    rec_img.save(output_image_path)
    print(f"Reconstructed image saved to {output_image_path}")


if __name__ == "__main__":
    checkpoints = ["epoch10", "epoch39", "epoch300"]
    for i in range(5):
        input_image_path = f"data/img{i}.png"
        for ckpt in checkpoints:
            output_image_path = f"data/img{i}_reconstructed_{ckpt}.png"
            checkpoint_path = f"mae_checkpoint_{ckpt}.pth"
            main(input_image_path, output_image_path, checkpoint_path)
