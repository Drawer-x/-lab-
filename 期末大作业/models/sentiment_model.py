"""
情感分析深度学习模型 - 基于BERT的中文情感分类
"""
import torch
import torch.nn as nn

# 检查是否有transformers库
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("警告: 未安装transformers库，将使用规则方法")


class BertSentimentModel:
    """基于预训练BERT的情感分析模型"""
    
    def __init__(self, model_name='uer/roberta-base-finetuned-jd-binary-chinese'):
        """
        使用已经在中文情感数据上微调过的模型
        模型来源: https://huggingface.co/uer/roberta-base-finetuned-jd-binary-chinese
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("需要安装transformers库")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"加载模型中... 使用设备: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("模型加载完成!")
    
    def predict(self, text, max_length=128):
        """预测单条文本的情感"""
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predicted = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted].item()
        
        return {
            'sentiment': predicted,
            'label': '正面' if predicted == 1 else '负面',
            'confidence': round(confidence, 2)
        }
    
    def predict_batch(self, texts, max_length=128):
        """批量预测"""
        return [self.predict(text, max_length) for text in texts]


class SentimentPredictor:
    """情感预测器 - 统一接口"""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if SentimentPredictor._model is None:
            self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if HAS_TRANSFORMERS:
            try:
                SentimentPredictor._model = BertSentimentModel()
                self.use_bert = True
                print("使用BERT深度学习模型")
            except Exception as e:
                print(f"BERT模型加载失败: {e}")
                print("降级使用规则方法")
                self.use_bert = False
        else:
            self.use_bert = False
            print("使用规则方法（未安装transformers）")
    
    def predict(self, text):
        """预测情感"""
        if self.use_bert and SentimentPredictor._model:
            return SentimentPredictor._model.predict(text)
        else:
            return self._rule_based_predict(text)
    
    def _rule_based_predict(self, text):
        """基于规则的情感分析（备用）"""
        strong_pos = ['强烈推荐', '太棒了', '太好了', '非常棒', '非常好', '很棒', 
                      '超级棒', '神作', '佳作', '必看', '五星', '满分', '完美']
        strong_neg = ['太差了', '太烂了', '垃圾', '烂片', '差劲', '浪费时间',
                      '不推荐', '后悔', '难看', '无聊透顶', '太失望']
        
        for p in strong_pos:
            if p in text:
                return {'sentiment': 1, 'label': '正面', 'confidence': 0.92}
        for n in strong_neg:
            if n in text:
                return {'sentiment': 0, 'label': '负面', 'confidence': 0.92}
        
        pos_words = ['好', '棒', '赞', '精彩', '感动', '推荐', '喜欢', '不错', '值得']
        neg_words = ['差', '烂', '糟', '失望', '无聊', '难看', '坑', '一般']
        
        pos = sum(1 for w in pos_words if w in text)
        neg = sum(1 for w in neg_words if w in text)
        
        if pos > neg:
            return {'sentiment': 1, 'label': '正面', 'confidence': 0.7}
        elif neg > pos:
            return {'sentiment': 0, 'label': '负面', 'confidence': 0.7}
        return {'sentiment': 1, 'label': '正面', 'confidence': 0.55}


# LSTM模型定义（用于展示深度学习架构）
class LSTMSentimentModel(nn.Module):
    """基于LSTM的情感分类模型"""
    
    def __init__(self, vocab_size=50000, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMSentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.dropout(hidden_cat)
        return self.fc(output)


if __name__ == '__main__':
    print("测试情感分析模型...")
    
    predictor = SentimentPredictor()
    
    test_texts = [
        "这部电影太棒了，强烈推荐！",
        "剧情精彩，演员演技在线",
        "太差劲了，浪费时间",
        "无聊透顶，看了想睡觉",
        "还行吧，一般般"
    ]
    
    print("\n测试结果:")
    for text in test_texts:
        result = predictor.predict(text)
        print(f"  {text}")
        print(f"    -> {result['label']} (置信度: {result['confidence']:.0%})")
