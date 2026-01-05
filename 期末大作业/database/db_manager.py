"""
数据库管理模块 - 使用SQLAlchemy管理电影评论数据
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class MovieReview(Base):
    """电影评论数据表"""
    __tablename__ = 'movie_reviews'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    movie_name = Column(String(200), nullable=False)
    review_text = Column(Text, nullable=False)
    sentiment = Column(Integer)  # 0: 负面, 1: 正面
    predicted_sentiment = Column(Integer)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self):
        return {
            'id': self.id,
            'movie_name': self.movie_name,
            'review_text': self.review_text,
            'sentiment': self.sentiment,
            'predicted_sentiment': self.predicted_sentiment,
            'confidence': self.confidence,
            'created_at': str(self.created_at)
        }

class DatabaseManager:
    """数据库管理类"""
    
    def __init__(self, db_path='sqlite:///movie_reviews.db'):
        self.engine = create_engine(db_path, echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def add_review(self, movie_name, review_text, sentiment=None):
        """添加评论"""
        review = MovieReview(
            movie_name=movie_name,
            review_text=review_text,
            sentiment=sentiment
        )
        self.session.add(review)
        self.session.commit()
        return review.id

    
    def add_reviews_batch(self, reviews_data):
        """批量添加评论"""
        for data in reviews_data:
            review = MovieReview(**data)
            self.session.add(review)
        self.session.commit()
    
    def get_all_reviews(self):
        """获取所有评论"""
        return self.session.query(MovieReview).all()
    
    def get_reviews_by_movie(self, movie_name):
        """按电影名获取评论"""
        return self.session.query(MovieReview).filter(
            MovieReview.movie_name.like(f'%{movie_name}%')
        ).all()
    
    def get_reviews_by_sentiment(self, sentiment):
        """按情感标签获取评论"""
        return self.session.query(MovieReview).filter(
            MovieReview.sentiment == sentiment
        ).all()
    
    def update_prediction(self, review_id, predicted_sentiment, confidence):
        """更新预测结果"""
        review = self.session.query(MovieReview).filter(
            MovieReview.id == review_id
        ).first()
        if review:
            review.predicted_sentiment = predicted_sentiment
            review.confidence = confidence
            self.session.commit()
    
    def get_statistics(self):
        """获取统计信息"""
        total = self.session.query(MovieReview).count()
        # 使用predicted_sentiment统计预测结果
        positive = self.session.query(MovieReview).filter(
            MovieReview.predicted_sentiment == 1
        ).count()
        negative = self.session.query(MovieReview).filter(
            MovieReview.predicted_sentiment == 0
        ).count()
        return {
            'total': total,
            'positive': positive,
            'negative': negative,
            'positive_ratio': positive / total if total > 0 else 0
        }
    
    def clear_all(self):
        """清空所有数据"""
        self.session.query(MovieReview).delete()
        self.session.commit()

    def close(self):
        """关闭数据库连接"""
        self.session.close()


if __name__ == '__main__':
    # 测试数据库功能
    db = DatabaseManager()
    
    # 添加测试数据
    db.add_review('流浪地球', '非常震撼的科幻大片，特效一流！', 1)
    db.add_review('流浪地球', '剧情有些拖沓，但整体还不错', 1)
    db.add_review('某烂片', '浪费时间，不推荐观看', 0)
    
    # 获取统计
    stats = db.get_statistics()
    print(f"数据库统计: {stats}")
    
    db.close()
