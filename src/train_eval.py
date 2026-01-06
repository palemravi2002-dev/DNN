# src/train_eval.py
import torch
import torch.nn as nn
from torch.optim import Adam
from nltk.translate.bleu_score import sentence_bleu

from model import MultimodalStoryModel
from data import get_dataloader

class Trainer:
    def __init__(self, lr=1e-4, epochs=1, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = MultimodalStoryModel().to(device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self):
        loader = get_dataloader(batch_size=2, max_stories=1000)
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            for batch in loader:
                imgs = batch["context_images"].to(self.device)
                txts = batch["context_text"]
                tgt_txt = batch["target_text"]

                logits = self.model(imgs, txts)  # (B,1,vocab)

                logits = logits.squeeze(1)  # (B,vocab)
                # Dummy target token: this baseline predicts CLS token (id=101)
                target = torch.ones(logits.size(0), dtype=torch.long).to(self.device)

                loss = self.loss_fn(logits, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print("Loss:", loss.item())

    def evaluate(self, test_samples=20):
        loader = get_dataloader(batch_size=1, max_stories=200)
        bleu_scores = []

        for i, batch in enumerate(loader):
            if i >= test_samples:
                break

            imgs = batch["context_images"].to(self.device)
            txts = batch["context_text"]
            tgt_txt = batch["target_text"][0]

            logits = self.model(imgs, txts)
            pred_id = logits.argmax(-1).item()

            pred_word = f"token_{pred_id}"  # minimal baseline
            bleu = sentence_bleu([tgt_txt.split()], pred_word.split())
            bleu_scores.append(bleu)

        print("Average BLEU:", sum(bleu_scores)/len(bleu_scores))
