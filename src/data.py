# src/data.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


class StoryDataset(Dataset):
    def __init__(self, seq_len=4, image_size=224, max_stories=2000):
        self.seq_len = seq_len
        self.context_len = seq_len - 1

        print("Loading dataset...")
        self.ds = load_dataset("MAPLE-WestLake-AIGC/OpenstoryPlusPlus", split="train")

        print("Grouping by story...")
        self.stories = {}
        for row in self.ds:
            vid = row["video_id"]
            if vid not in self.stories:
                self.stories[vid] = []
            self.stories[vid].append({
                "image": row["png"],
                "caption": row["json"]["caption"],
                "timestamp": self.parse_timestamp(row["time_stamp"])
            })

        # limit number for speed
        keys = list(self.stories.keys())[:max_stories]
        self.stories = {k: self.stories[k] for k in keys}

        print("Sorting...")
        for vid in self.stories:
            self.stories[vid] = sorted(self.stories[vid], key=lambda x: x["timestamp"])

        print("Building sequences...")
        self.samples = self.build_sequences(self.stories)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def parse_timestamp(self, stamp):
        return tuple(int(x) for x in stamp.split("-"))

    def build_sequences(self, stories):
        seqs = []
        for vid, frames in stories.items():
            if len(frames) < self.seq_len:
                continue
            for i in range(len(frames) - self.seq_len + 1):
                seqs.append(frames[i:i+self.seq_len])
        return seqs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]

        ctx_imgs, ctx_txts = [], []
        for item in seq[:-1]:
            img = Image.fromarray(item["image"])
            ctx_imgs.append(self.transform(img))
            ctx_txts.append(item["caption"])

        tgt = seq[-1]
        timg = Image.fromarray(tgt["image"])
        tgt_img = self.transform(timg)
        tgt_txt = tgt["caption"]

        return {
            "context_images": torch.stack(ctx_imgs),
            "context_text": ctx_txts,
            "target_image": tgt_img,
            "target_text": tgt_txt
        }


def get_dataloader(batch_size=4, seq_len=4, image_size=224, max_stories=1000):
    dataset = StoryDataset(seq_len=seq_len, image_size=image_size, max_stories=max_stories)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
