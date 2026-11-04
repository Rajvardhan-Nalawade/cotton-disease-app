import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import io
import timm
from einops import rearrange
from einops.layers.torch import Rearrange
from huggingface_hub import hf_hub_download
import logging
import timm
from PIL import Image, ImageOps


logger = logging.getLogger("cotton")
logging.basicConfig(level=logging.INFO)

GATE_THRESHOLD = 0.3     # reject only if P(cotton) is below this
VIT_CONF_FLOOR = 0.6     # below this, say "uncertain"

CLASS_NAMES = [
    'Bacterial Blight', 'Curl Virus', 'Healthy Leaf',
    'Herbicide Growth Damage', 'Leaf Hopper Jassids',
    'Leaf Redding', 'Leaf Variegation'
]

MODEL_PATH = hf_hub_download(
    repo_id="Raj-Nal/cotton-vit-model",
    filename="cotton_vit_eval.pth"
)

GATE_PATH = hf_hub_download(
    repo_id="Raj-Nal/cotton-vit-model",
    filename="gate_v3.pt"
)


#Model

class RobustEdgeDetectionModule(nn.Module):
    def __init__(self, mode='sobel'):
        super().__init__()
        self.mode = mode

        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)

        self.register_buffer('sobel_x_kernel', sobel_x.view(1,1,3,3).repeat(3,1,1,1))
        self.register_buffer('sobel_y_kernel', sobel_y.view(1,1,3,3).repeat(3,1,1,1))

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x_kernel, padding=1, groups=3)
        grad_y = F.conv2d(x, self.sobel_y_kernel, padding=1, groups=3)
        edge_map = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return edge_map / (edge_map.max() + 1e-6)

class SafeCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context):
        qkv = self.to_qkv(torch.cat([x, context], dim=1)).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class RobustAttentionGuidedEdgeViT(nn.Module):
    def __init__(self, num_classes=7, edge_mode='sobel', pretrained=False):
        super().__init__()
        self.edge_detector = RobustEdgeDetectionModule(mode=edge_mode)
        self.vit = timm.create_model('deit_tiny_patch16_224', pretrained=pretrained)
        self.embed_dim = self.vit.embed_dim
        self.vit.head = nn.Identity()

        self.edge_patch_embed = nn.Sequential(
            nn.Conv2d(3, self.embed_dim, 16, 16),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(self.embed_dim)
        )

        self.cross_attn = SafeCrossAttention(self.embed_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

        num_patches = (224 // 16) ** 2
        self.edge_pos_embed = nn.Parameter(torch.randn(1, num_patches, self.embed_dim) * 0.02)

    def forward(self, x):
        edge_maps = self.edge_detector(x)
        rgb_features = self.vit.forward_features(x)
        edge_tokens = self.edge_patch_embed(edge_maps) + self.edge_pos_embed

        cls_token = rgb_features[:, 0:1]
        patch_tokens = rgb_features[:, 1:]
        attended = self.cross_attn(patch_tokens, edge_tokens)

        combined = torch.cat([cls_token, attended], dim=1)
        logits = self.classifier(combined[:, 0])
        return logits

#Load model

def load_model(weights_path):
    model = RobustAttentionGuidedEdgeViT()
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

model = load_model(MODEL_PATH)

#Load gate
def load_gate(path):
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m

gate = load_gate(GATE_PATH)

#Preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

#Prediction

def predict_image_bytes(image_bytes):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        p_cotton = F.softmax(gate(tensor), dim=1)[0, 0].item()

    if p_cotton < GATE_THRESHOLD:
        logger.info("status=rejected p_cotton=%.3f", p_cotton)
        return {"status": "rejected", "reason": "not_cotton", "p_cotton": p_cotton}

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    idx = int(torch.argmax(probs).item())
    conf = float(probs[idx])

    status = "ok" if conf >= VIT_CONF_FLOOR else "uncertain"
    logger.info("status=%s p_cotton=%.3f class=%s conf=%.3f",
                status, p_cotton, CLASS_NAMES[idx], conf)

    return {
        "status": status,
        "class": CLASS_NAMES[idx],
        "confidence": conf,
        "p_cotton": p_cotton,
    }