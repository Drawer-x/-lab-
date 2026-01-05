"""
示例数据生成模块 - 生成用于演示的电影评论数据
"""
import pandas as pd
import random

# 正面评论模板
POSITIVE_TEMPLATES = [
    "这部电影太棒了，{aspect}非常出色！",
    "{movie}真的很好看，{aspect}让人印象深刻",
    "强烈推荐{movie}，{aspect}一流",
    "看完{movie}感觉很震撼，{aspect}太赞了",
    "{movie}是今年最好的电影之一，{aspect}无可挑剔",
    "五星好评！{movie}的{aspect}让我感动",
    "{movie}值得一看，{aspect}处理得很好",
    "被{movie}圈粉了，{aspect}太棒了",
    "{movie}超出预期，{aspect}令人惊喜",
    "必看佳作！{movie}的{aspect}堪称完美"
]

# 负面评论模板
NEGATIVE_TEMPLATES = [
    "这部电影太差了，{aspect}简直是灾难",
    "{movie}让人失望，{aspect}一塌糊涂",
    "不推荐{movie}，{aspect}太烂了",
    "浪费时间看{movie}，{aspect}毫无亮点",
    "{movie}是今年最烂的电影，{aspect}不忍直视",
    "一星差评！{movie}的{aspect}让人崩溃",
    "{movie}不值得看，{aspect}处理得很差",
    "被{movie}劝退了，{aspect}太糟糕了",
    "{movie}远低于预期，{aspect}令人失望",
    "烂片预警！{movie}的{aspect}简直是笑话"
]

# 电影名称
MOVIES = [
    "流浪地球", "战狼2", "哪吒之魔童降世", "红海行动", "唐人街探案3",
    "你好李焕英", "长津湖", "我不是药神", "西虹市首富", "疯狂的外星人",
    "飞驰人生", "独行月球", "满江红", "消失的她", "孤注一掷"
]

# 评价方面
ASPECTS = [
    "剧情", "演技", "特效", "配乐", "摄影", "节奏", "台词", "情感表达",
    "人物塑造", "故事结构", "视觉效果", "导演功力"
]

def generate_review(sentiment):
    """生成单条评论"""
    movie = random.choice(MOVIES)
    aspect = random.choice(ASPECTS)
    
    if sentiment == 1:
        template = random.choice(POSITIVE_TEMPLATES)
    else:
        template = random.choice(NEGATIVE_TEMPLATES)
    
    review = template.format(movie=movie, aspect=aspect)
    return movie, review, sentiment

def generate_dataset(num_samples=1000, positive_ratio=0.5):
    """生成数据集"""
    data = []
    num_positive = int(num_samples * positive_ratio)
    
    for _ in range(num_positive):
        movie, review, sentiment = generate_review(1)
        data.append({'movie_name': movie, 'review_text': review, 'sentiment': sentiment})
    
    for _ in range(num_samples - num_positive):
        movie, review, sentiment = generate_review(0)
        data.append({'movie_name': movie, 'review_text': review, 'sentiment': sentiment})
    
    random.shuffle(data)
    return pd.DataFrame(data)

def save_dataset(df, filepath='movie_reviews.csv'):
    """保存数据集"""
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"数据集已保存到 {filepath}")
    print(f"总样本数: {len(df)}")
    print(f"正面评论: {len(df[df['sentiment']==1])}")
    print(f"负面评论: {len(df[df['sentiment']==0])}")


if __name__ == '__main__':
    # 生成示例数据集
    df = generate_dataset(num_samples=2000, positive_ratio=0.5)
    save_dataset(df, 'movie_reviews.csv')
    
    # 显示样例
    print("\n数据样例:")
    print(df.head(10).to_string())
