import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.projection = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        # x shape: (B, embed_dim, H_p, W_p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # qkv: (B, N, 3 * C) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class FFN(nn.Module):
    """MMPose FFN representation matching layers.0.0 and layers.1 keys"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU()
            ),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.layers(x)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        
        self.layers = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.ln1 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.ln1(x)
        return x

class TopdownHeatmapSimpleHead(nn.Module):
    def __init__(self, in_channels, out_channels=17):
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.final_layer = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x is (B, num_patches, embed_dim) from ViT
        B, N, C = x.shape
        H_p, W_p = 16, 12  # For 256x192 input image
        x = x.transpose(1, 2).reshape(B, C, H_p, W_p)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

class ViTPose(nn.Module):
    def __init__(self, img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=384, depth=12, num_heads=12, out_channels=17):
        super().__init__()
        self.backbone = VisionTransformer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                                          embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        self.head = TopdownHeatmapSimpleHead(in_channels=embed_dim, out_channels=out_channels)

    def forward(self, x):
        features = self.backbone(x)
        heatmaps = self.head(features)
        return heatmaps

def load_vitpose_model(weight_path, device="cpu"):
    """Loads the ViTPose-s model with pretrained weights"""
    model = ViTPose(img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=384, depth=12, num_heads=12, out_channels=17)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Strip prefixes if checkpoint is saved inside mmpose config structure
    # Standard format has 'backbone.pos_embed' etc. which matches our class structure.
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
