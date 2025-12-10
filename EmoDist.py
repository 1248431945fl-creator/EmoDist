# EmoDist.py  --- 完整修正版，可直接运行，无报错
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from math import floor

# ============================================================
#                 Dataset + Collate_fn
# ============================================================
class MultiModalDataset(Dataset):
    def __init__(self, fmri_np, audio_np, labels_np, emotion_labels=None):
        """
        fmri_np: (N,90,90)
        audio_np: (N,64,640)
        labels_np: (N,)
        emotion_labels: list of lists OR None
        """
        self.fmri = fmri_np.astype(np.float32)
        self.audio = audio_np.astype(np.float32)
        self.labels = labels_np.astype(np.float32)
        self.emotion_labels = emotion_labels  # may be None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fmri = torch.tensor(self.fmri[idx]).unsqueeze(0)    # (1,90,90)
        audio = torch.tensor(self.audio[idx]).unsqueeze(0)  # (1,64,640)
        label = torch.tensor(self.labels[idx]).float()

        # -------- FIX: NEVER RETURN None ----------
        if self.emotion_labels is None:
            emo = []      # empty list, safe for DataLoader
        else:
            emo = self.emotion_labels[idx]

        return fmri, audio, label, emo


# collate_fn that supports "emo = []"
def collate_fn(batch):
    fmri, audio, label, emo = zip(*batch)

    fmri = torch.stack(fmri)
    audio = torch.stack(audio)
    label = torch.stack(label)
    emo = list(emo)  # keep as python list

    return fmri, audio, label, emo


# ============================================================
#                     FMRI Encoder
# ============================================================
class FMRIEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,5,2,2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        h = self.net(x).view(x.size(0), -1)
        return self.fc(h)


# ============================================================
#                     Audio Encoder
# ============================================================
class AudioEncoder(nn.Module):
    def __init__(self, out_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,(3,5),(1,2),(1,2)), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16,32,(3,5),(2,2),(1,2)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,64,(3,5),(2,2),(1,2)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        h = self.net(x).view(x.size(0), -1)
        return self.fc(h)


# ============================================================
#               Emotion Encoding (EE)
# ============================================================
EMO_LOOKUP = {
    'disgust':0.1, 'fear':0.2, 'sadness':0.3, 'anger':0.4,
    'neutral':0.5, 'joy':0.9, 'surprise':1.0
}

def build_emotion_code(seq, de=128):
    """If seq=[], output neutral code."""
    if seq is None or len(seq)==0:
        return np.full((de,), 0.5, dtype=np.float32)

    values = [EMO_LOOKUP.get(e.lower(),0.5) for e in seq]

    wjs = sorted(set(EMO_LOOKUP.values()))
    N = len(values)

    pj = [sum(1 for v in values if abs(v-w)<1e-6) / N for w in wjs]
    ks = [int(floor(de*p)) for p in pj]

    code = []
    for w,k in zip(wjs,ks):
        code.extend([w]*k)

    if len(code)<de:
        code.extend([0.5]*(de - len(code)))

    return np.array(code[:de], dtype=np.float32)


# ============================================================
#                   EWA Fusion Module
# ============================================================
class EWAFusion(nn.Module):
    def __init__(self, feat_dim, hidden=1024):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, feat_dim)

    def forward(self, x):
        alpha = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
        return x * alpha


# ============================================================
#              Full Multi-modal Fusion Model
# ============================================================
class MultiModalCore(nn.Module):
    def __init__(self, fmri_dim=256, audio_dim=768, emo_dim=128):
        super().__init__()
        self.fmri_enc = FMRIEncoder(fmri_dim)
        self.audio_enc = AudioEncoder(audio_dim)
        self.emo_dim = emo_dim

        self.feat_dim = fmri_dim + audio_dim
        self.ewa = EWAFusion(self.feat_dim)

        self.clf = nn.Sequential(
            nn.Linear(self.feat_dim + emo_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, fmri, audio, emotion_seq_batch):
        b = fmri.size(0)
        f_fmri = self.fmri_enc(fmri)
        f_audio = self.audio_enc(audio)

        concat = torch.cat([f_fmri, f_audio], dim=1)
        ci_f = self.ewa(concat)

        # -------- FIX: empty emotion list OR None → neutral code ----------
        if (emotion_seq_batch is None) or (len(emotion_seq_batch[0])==0):
            ci_e = np.tile(build_emotion_code([], de=self.emo_dim), (b,1))
        else:
            ci_e = np.stack([build_emotion_code(seq, self.emo_dim)
                             for seq in emotion_seq_batch], axis=0)

        ci_e = torch.tensor(ci_e, dtype=ci_f.dtype, device=ci_f.device)

        fused = torch.cat([ci_e, ci_f], dim=1)
        logits = self.clf(fused).squeeze(1)
        return logits


# ============================================================
#                       Train & Eval
# ============================================================
def train_one_epoch(model, loader, optim, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total = 0
    for fmri, audio, label, emo in loader:
        fmri, audio, label = fmri.to(device), audio.to(device), label.to(device)

        logits = model(fmri, audio, emo)
        loss = loss_fn(logits, label)

        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item() * fmri.size(0)
    return total / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for fmri, audio, label, emo in loader:
            fmri, audio = fmri.to(device), audio.to(device)
            logit = model(fmri, audio, emo)
            prob = torch.sigmoid(logit).cpu().numpy()

            preds.append(prob)
            trues.append(label.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    acc = ((preds>=0.5).astype(int) == trues).mean()
    return {"acc": acc}


# ============================================================
#                   Main Script
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri_np", default="./MODMApcc.npy")
    parser.add_argument("--audio_mat", default="./audio_features.mat")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -------- load data --------
    fmri_np = np.load(args.fmri_np)                     # (N,90,90)
    audio_data = loadmat(args.audio_mat)
    audio_np = audio_data["stft_data"]                  # (N,64,640)
    labels_np = audio_data["label"].reshape(-1)         # (N,)

    # no emotion labels -> system will generate neutral code
    emotion_labels = None

    ds = MultiModalDataset(fmri_np, audio_np, labels_np, emotion_labels)
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )

    # -------- build model --------
    model = MultiModalCore().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    # -------- training --------
    for ep in range(1, args.epochs+1):
        loss = train_one_epoch(model, loader, optim, device)
        metric = evaluate(model, loader, device)
        print(f"[Epoch {ep}] loss={loss:.4f}  acc={metric['acc']:.4f}")

    torch.save(model.state_dict(), "multimodal_core_fixed.pth")
    print("Model saved to multimodal_core_fixed.pth")
