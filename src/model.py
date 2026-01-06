# src/model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18
from transformers import AutoTokenizer, AutoModel

class VisualEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        base = resnet18(weights=None)  # university-safe, no download needed
        base.fc = nn.Linear(base.fc.in_features, embed_dim)
        self.encoder = base

    def forward(self, x):
        return self.encoder(x)  # (B, 256)


class TextEncoder(nn.Module):
    """
    We use a small Transformer encoder via HuggingFace.
    """
    def __init__(self, model_name="distilbert-base-uncased", embed_dim=256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(768, embed_dim)

    def forward(self, texts):
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.encoder(**tokens)
        cls = outputs.last_hidden_state[:,0,:]  # (B,768)
        return self.proj(cls)  # (B,256)


class SequenceModel(nn.Module):
    def __init__(self, embed_dim=256, hidden=256):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)

    def forward(self, seq):
        out, _ = self.lstm(seq)
        return out[:, -1, :]  # last hidden state


class TextDecoder(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=256, hidden=256):
        super().__init__()
        self.decoder = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, embeds):
        out, _ = self.decoder(embeds)
        return self.out(out)  # logits


class MultimodalStoryModel(nn.Module):
    """
    Full baseline model: image encoder + text encoder + fusion + LSTM + text decoder.
    """
    def __init__(self):
        super().__init__()
        self.img_encoder = VisualEncoder()
        self.text_encoder = TextEncoder()

        self.fusion_dim = 256 + 256
        self.seq_model = SequenceModel(embed_dim=self.fusion_dim)

        self.decoder = TextDecoder()

    def forward(self, images, texts):
        """
        images: (B, K-1, 3, H, W)
        texts:  list of lists (B x (K-1 captions))
        """
        B, K1, _, _, _ = images.shape

        fused = []
        for i in range(K1):
            img_feat = self.img_encoder(images[:, i])
            txt_feat = self.text_encoder([t[i] for t in texts])  # t is list of captions per batch
            fused.append(torch.cat([img_feat, txt_feat], dim=1))

        fused = torch.stack(fused, dim=1)  # (B, K-1, fusion_dim)

        seq_embedding = self.seq_model(fused)  # (B, hidden)

        # For baseline: project seq_embedding to decoder input shape
        seq_embedding = seq_embedding.unsqueeze(1)  
        logits = self.decoder(seq_embedding)

        return logits  # (B,1,vocab_size)
