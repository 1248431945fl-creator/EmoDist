import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import models
import warnings

warnings.filterwarnings('ignore')


# 自定义数据集类
class MultiModalDepressionDataset(Dataset):
    def __init__(self, fmri_data, audio_data, labels):
        self.fmri_data = fmri_data  # shape: [N, 90, 90]
        self.audio_data = audio_data  # shape: [N, 64, 640]
        self.labels = labels  # shape: [N]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 确保数据是float32类型
        fmri = self.fmri_data[idx].float()
        audio = self.audio_data[idx].float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # 为fMRI和音频数据添加通道维度
        fmri = fmri.unsqueeze(0)  # [1, 90, 90]
        audio = audio.unsqueeze(0)  # [1, 64, 640]

        return fmri, audio, label


# 修改DenseNet121以接受单通道输入
class ModifiedDenseNet121(nn.Module):
    def __init__(self, pretrained=True):
        super(ModifiedDenseNet121, self).__init__()
        # 加载预训练的DenseNet121
        densenet = models.densenet121(pretrained=pretrained)

        # 修改第一层卷积以接受单通道输入
        original_first_conv = densenet.features[0]
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(densenet.features)[1:]
        )

        # 获取分类层之前的特征维度
        self.num_features = densenet.classifier.in_features

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


# 多模态融合模型
class MultiModalDepressionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MultiModalDepressionModel, self).__init__()

        # fMRI分支 (DenseNet121)
        self.fmri_backbone = ModifiedDenseNet121(pretrained=True)

        # 音频分支 (DenseNet121)
        self.audio_backbone = ModifiedDenseNet121(pretrained=True)

        # 获取特征维度
        fmri_feature_dim = self.fmri_backbone.num_features  # 1024
        audio_feature_dim = self.audio_backbone.num_features  # 1024

        # 特征融合前的维度调整
        self.fmri_fc = nn.Sequential(
            nn.Linear(fmri_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.audio_fc = nn.Sequential(
            nn.Linear(audio_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # 注意力机制（元素级注意力）
        self.element_wise_attention = nn.Sequential(
            nn.Linear(1024, 1024),  # 输入是拼接后的特征
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.Sigmoid()
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, fmri, audio):
        # 提取fMRI特征
        fmri_features = self.fmri_backbone(fmri)
        fmri_features = self.fmri_fc(fmri_features)

        # 提取音频特征
        audio_features = self.audio_backbone(audio)
        audio_features = self.audio_fc(audio_features)

        # 特征拼接
        combined_features = torch.cat([fmri_features, audio_features], dim=1)

        # 元素级注意力
        attention_weights = self.element_wise_attention(combined_features)
        attended_features = combined_features * attention_weights

        # 分类
        output = self.classifier(attended_features)

        return output, attention_weights


# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for fmri, audio, labels in train_loader:
            fmri, audio, labels = fmri.to(device), audio.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(fmri, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for fmri, audio, labels in val_loader:
                fmri, audio, labels = fmri.to(device), audio.to(device), labels.to(device)

                outputs, _ = model(fmri, audio)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 调整学习率
        scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    return model, train_losses, val_losses, train_accs, val_accs


# 评估函数
def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_preds = []
    all_labels = []
    all_attention_weights = []

    with torch.no_grad():
        for fmri, audio, labels in test_loader:
            fmri, audio, labels = fmri.to(device), audio.to(device), labels.to(device)

            outputs, attention_weights = model(fmri, audio)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attention_weights.append(attention_weights.cpu().numpy())

    # 计算评估指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{cm}')

    return accuracy, precision, recall, f1, cm, all_attention_weights


# 主函数
def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载数据
    print("Loading data...")
    audio_data = loadmat('./audio_features.mat')
    train_data_ = np.load('./MODMApcc.npy')

    # 提取音频特征和标签
    audio_feature, train_label = audio_data['stft_data'], audio_data['label']
    train_label = train_label.T

    # 转换为tensor
    train_data = torch.tensor(train_data_, dtype=torch.float32)  # fMRI数据
    audio_feature = torch.tensor(audio_feature, dtype=torch.float32)  # 音频数据
    train_label = torch.tensor(train_label.flatten(), dtype=torch.long)  # 标签

    # 确保数据维度匹配
    print(f"fMRI data shape: {train_data.shape}")
    print(f"Audio data shape: {audio_feature.shape}")
    print(f"Labels shape: {train_label.shape}")

    # 分割数据集
    dataset_size = len(train_data)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    indices = torch.randperm(dataset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 创建数据集
    train_dataset = MultiModalDepressionDataset(
        train_data[train_indices],
        audio_feature[train_indices],
        train_label[train_indices]
    )

    val_dataset = MultiModalDepressionDataset(
        train_data[val_indices],
        audio_feature[val_indices],
        train_label[val_indices]
    )

    test_dataset = MultiModalDepressionDataset(
        train_data[test_indices],
        audio_feature[test_indices],
        train_label[test_indices]
    )

    # 创建数据加载器
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # 创建模型
    print("Creating model...")
    model = MultiModalDepressionModel(num_classes=2)

    # 训练模型
    print("Training model...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs=30, lr=0.001
    )

    # 评估模型
    print("Evaluating model...")
    accuracy, precision, recall, f1, cm, attention_weights = evaluate_model(model, test_loader)

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    }, 'multimodal_depression_model.pth')

    print("Model saved to multimodal_depression_model.pth")

    return model, train_losses, val_losses, train_accs, val_accs


if __name__ == "__main__":
    model, train_losses, val_losses, train_accs, val_accs = main()