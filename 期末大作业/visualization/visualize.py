"""
数据可视化模块 - 生成分析图表
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
import sys
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_sentiment_distribution(df, save_path='sentiment_distribution.png'):
    """绘制情感分布饼图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 饼图
    sentiment_counts = df['sentiment'].value_counts()
    labels = ['正面评论', '负面评论']
    colors = ['#2ecc71', '#e74c3c']
    axes[0].pie(sentiment_counts, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=(0.05, 0))
    axes[0].set_title('评论情感分布', fontsize=14)
    
    # 柱状图
    sns.countplot(data=df, x='sentiment', palette=['#e74c3c', '#2ecc71'], ax=axes[1])
    axes[1].set_xticklabels(['负面', '正面'])
    axes[1].set_xlabel('情感类别')
    axes[1].set_ylabel('评论数量')
    axes[1].set_title('评论数量统计', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"情感分布图已保存到 {save_path}")

def plot_movie_sentiment(df, top_n=10, save_path='movie_sentiment.png'):
    """绘制各电影情感分析"""
    movie_sentiment = df.groupby('movie_name')['sentiment'].agg(['mean', 'count'])
    movie_sentiment = movie_sentiment.sort_values('count', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(movie_sentiment))
    bars = ax.bar(x, movie_sentiment['mean'], color='steelblue')
    
    # 根据情感值着色
    for i, bar in enumerate(bars):
        if movie_sentiment['mean'].iloc[i] >= 0.6:
            bar.set_color('#2ecc71')
        elif movie_sentiment['mean'].iloc[i] <= 0.4:
            bar.set_color('#e74c3c')
        else:
            bar.set_color('#f39c12')
    
    ax.set_xticks(x)
    ax.set_xticklabels(movie_sentiment.index, rotation=45, ha='right')
    ax.set_ylabel('正面评论比例')
    ax.set_title(f'Top {top_n} 电影情感分析', fontsize=14)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"电影情感分析图已保存到 {save_path}")


def plot_review_length_distribution(df, save_path='review_length.png'):
    """绘制评论长度分布"""
    df['review_length'] = df['review_text'].str.len()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 整体分布
    sns.histplot(df['review_length'], bins=30, kde=True, ax=axes[0], color='steelblue')
    axes[0].set_xlabel('评论长度（字符数）')
    axes[0].set_ylabel('频数')
    axes[0].set_title('评论长度分布', fontsize=14)
    
    # 按情感分组
    sns.boxplot(data=df, x='sentiment', y='review_length', palette=['#e74c3c', '#2ecc71'], ax=axes[1])
    axes[1].set_xticklabels(['负面', '正面'])
    axes[1].set_xlabel('情感类别')
    axes[1].set_ylabel('评论长度')
    axes[1].set_title('不同情感评论长度对比', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"评论长度分布图已保存到 {save_path}")

def plot_training_history(history, save_path='training_history.png'):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 损失曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='训练损失')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练/验证损失曲线', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', label='训练准确率')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='验证准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('训练/验证准确率曲线', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练历史图已保存到 {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title('混淆矩阵', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵图已保存到 {save_path}")

def generate_all_visualizations(df, output_dir='./figures'):
    """生成所有可视化图表"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    plot_sentiment_distribution(df, f'{output_dir}/sentiment_distribution.png')
    plot_movie_sentiment(df, save_path=f'{output_dir}/movie_sentiment.png')
    plot_review_length_distribution(df, f'{output_dir}/review_length.png')
    
    print(f"\n所有图表已保存到 {output_dir} 目录")


if __name__ == '__main__':
    # 测试可视化
    from data.sample_data import generate_dataset
    
    df = generate_dataset(1000)
    generate_all_visualizations(df)
