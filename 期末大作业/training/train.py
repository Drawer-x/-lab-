"""
模型训练模块 - 训练情感分析模型
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.sentiment_model import SentimentClassifier

class ReviewDataset(Dataset):
    """电影评论数据集"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class Trainer:
    """模型训练器"""
    
    def __init__(self, model, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
    def train_epoch(self, dataloader, optimizer, scheduler, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        predictions, true_labels = [], []
        
        progress_bar = tqdm(dataloader, desc='Training')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        accuracy = accuracy_score(true_labels, predictions)
        return total_loss / len(dataloader), accuracy
    
    def evaluate(self, dataloader, criterion):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


def train_model(train_texts, train_labels, val_texts, val_labels,
                epochs=3, batch_size=16, learning_rate=2e-5, max_length=128):
    """完整的训练流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = SentimentClassifier()
    
    # 创建数据集
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    trainer = Trainer(model, device)
    best_f1 = 0
    
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print('='*50)
        
        train_loss, train_acc = trainer.train_epoch(
            train_loader, optimizer, scheduler, criterion
        )
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        
        val_metrics = trainer.evaluate(val_loader, criterion)
        print(f"验证损失: {val_metrics['loss']:.4f}")
        print(f"验证准确率: {val_metrics['accuracy']:.4f}")
        print(f"验证F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'best_model.pth')
            print("保存最佳模型!")
    
    return model


if __name__ == '__main__':
    # 示例：使用模拟数据测试训练流程
    print("准备测试数据...")
    
    # 模拟数据
    sample_texts = [
        "这部电影太棒了，强烈推荐！",
        "剧情精彩，演员演技在线",
        "浪费时间，不值得看",
        "特效一般，故事老套",
        "非常感人的电影，看哭了",
        "无聊透顶，看了一半就走了"
    ] * 10
    
    sample_labels = [1, 1, 0, 0, 1, 0] * 10
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        sample_texts, sample_labels, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")
