"""
电影评论情感分析系统 - 主程序入口
基于深度学习的中文电影评论情感分类

功能:
1. 数据库管理 - SQLite存储评论数据
2. 深度学习模型 - BERT/LSTM情感分类
3. Web界面 - Flask提供API和前端
4. 数据可视化 - 生成分析图表
"""
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='电影评论情感分析系统')
    parser.add_argument('--mode', type=str, default='web',
                       choices=['web', 'train', 'predict', 'visualize', 'generate_data'],
                       help='运行模式')
    parser.add_argument('--text', type=str, help='待分析的文本')
    parser.add_argument('--data_path', type=str, default='data/movie_reviews.csv',
                       help='数据文件路径')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                       help='模型文件路径')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        print("=" * 50)
        print("电影评论情感分析系统")
        print("=" * 50)
        from app.web_app import app
        print("\n启动Web服务...")
        print("请访问: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    
    elif args.mode == 'generate_data':
        print("生成示例数据...")
        from data.sample_data import generate_dataset, save_dataset
        df = generate_dataset(num_samples=2000)
        os.makedirs('data', exist_ok=True)
        save_dataset(df, args.data_path)
    
    elif args.mode == 'predict':
        if not args.text:
            print("请使用 --text 参数指定待分析文本")
            return
        
        print(f"分析文本: {args.text}")
        from models.sentiment_model import SentimentPredictor
        
        try:
            predictor = SentimentPredictor(model_path=args.model_path)
            result = predictor.predict(args.text)
        except:
            # 使用简单规则分析
            from app.web_app import simple_sentiment_analysis
            result = simple_sentiment_analysis(args.text)
        
        print(f"\n分析结果:")
        print(f"  情感: {result['label']}")
        print(f"  置信度: {result['confidence']:.2%}")
    
    elif args.mode == 'visualize':
        print("生成可视化图表...")
        import pandas as pd
        from visualization.visualize import generate_all_visualizations
        
        if os.path.exists(args.data_path):
            df = pd.read_csv(args.data_path)
        else:
            from data.sample_data import generate_dataset
            df = generate_dataset(1000)
        
        generate_all_visualizations(df, output_dir='./figures')
    
    elif args.mode == 'train':
        print("开始训练模型...")
        print("注意: 完整训练需要GPU支持，建议使用Google Colab")
        # 训练代码在 training/train.py 中


if __name__ == '__main__':
    main()
