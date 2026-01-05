"""
Flask Web应用 - 提供情感分析API和Web界面
"""
from flask import Flask, render_template, request, jsonify
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import DatabaseManager
from models.sentiment_model import SentimentPredictor

app = Flask(__name__, template_folder='templates', static_folder='static')
db = DatabaseManager()

# 全局预测器
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = SentimentPredictor()
    return predictor


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    movie_name = data.get('movie_name', '未知电影')

    if not text:
        return jsonify({'error': '请输入评论文本'}), 400

    pred = get_predictor()
    result = pred.predict(text)
    
    review_id = db.add_review(movie_name, text, result['sentiment'])
    db.update_prediction(review_id, result['sentiment'], result['confidence'])

    return jsonify({
        'review_id': review_id,
        'text': text,
        'sentiment': result['sentiment'],
        'label': result['label'],
        'confidence': result['confidence']
    })


@app.route('/api/reviews', methods=['GET'])
def get_reviews():
    reviews = db.get_all_reviews()
    return jsonify([r.to_dict() for r in reviews[-50:]])


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    return jsonify(db.get_statistics())


@app.route('/api/search', methods=['GET'])
def search_reviews():
    movie_name = request.args.get('movie', '')
    reviews = db.get_reviews_by_movie(movie_name) if movie_name else db.get_all_reviews()
    return jsonify([r.to_dict() for r in reviews])


@app.route('/api/clear', methods=['POST'])
def clear_data():
    db.clear_all()
    return jsonify({'success': True})


@app.route('/api/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    pred = get_predictor()
    return jsonify({
        'use_bert': pred.use_bert,
        'model_type': 'BERT深度学习模型' if pred.use_bert else '规则方法'
    })


if __name__ == '__main__':
    print("="*50)
    print("电影评论情感分析系统")
    print("="*50)
    
    # 预加载模型
    print("\n初始化模型...")
    get_predictor()
    
    print("\n启动Web服务...")
    print("访问 http://localhost:5000")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)
