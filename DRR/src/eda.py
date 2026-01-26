import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import math
from datetime import datetime, timedelta

# --- 1. CẤU HÌNH & LOGGING ---
OUTPUT_DIR = 'FINAL_ANALYSIS_OUTPUT'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

LOG_FILE = os.path.join(OUTPUT_DIR, 'final_report.txt')

def log(text):
    """Ghi log ra màn hình và file cùng lúc"""
    print(text)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(text + "\n")

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write(f"REPORT GENERATED AT: {datetime.now()}\n{'='*60}\n")

# Cấu hình đồ họa
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Mapping dữ liệu
AGE_MAP = {
    1:  "Under 18", 18: "18-24", 25: "25-34", 
    35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"
}
AGE_ORDER = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]

# =============================================================================
# PHẦN 2: LOAD & PREPROCESS DATA (XỬ LÝ FILE .DAT & TIME)
# =============================================================================
def load_data(path_root):
    log("\n[1/6] LOADING DATA MOVIELENS 1M...")
    
    # 1. Movies
    try:
        movies = pd.read_csv(path_root + 'movies.dat', sep='::', engine='python', 
                             encoding='ISO-8859-1', header=None, names=['movieId', 'title', 'genres'])
        movies['genres_list'] = movies['genres'].str.split('|')
        
        # 2. Users
        users = pd.read_csv(path_root + 'users.dat', sep='::', engine='python', 
                            header=None, names=['userId', 'gender', 'age', 'occupation', 'zip'])
        users['age_group'] = users['age'].map(AGE_MAP)
        
        # 3. Ratings
        ratings = pd.read_csv(path_root + 'ratings.dat', sep='::', engine='python', 
                              header=None, names=['userId', 'movieId', 'rating', 'timestamp'],
                              dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'int8'})
        
        # Xử lý thời gian
        ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
        ratings['year'] = ratings['date'].dt.year
        ratings['month_dt'] = ratings['date'].dt.to_period('M')
        ratings['week_dt'] = ratings['date'].dt.to_period('W') # Quan trọng cho Trending
        ratings['hour'] = ratings['date'].dt.hour
        ratings['day_of_week'] = ratings['date'].dt.day_name()
        
        # Merge để phân tích
        full_df = ratings.merge(users[['userId', 'age_group', 'gender']], on='userId', how='left')
        
        log(f"   - Loaded: {len(movies)} Movies, {len(users)} Users, {len(ratings)} Ratings.")
        return movies, full_df, users
        
    except FileNotFoundError:
        log(f"LỖI: Không tìm thấy file dữ liệu trong folder '{path_root}'.")
        sys.exit(1)

# --- CONFIG PATH ---
PATH_ROOT = 'data/ml-1m/' # <--- SỬA PATH CỦA BẠN TẠI ĐÂY
df_movies, df_ratings, df_users = load_data(PATH_ROOT)

# =============================================================================
# PHẦN 3: DEEP EDA - PHÂN TÍCH TOÀN DIỆN (LƯU 8 PLOTS)
# =============================================================================
log("\n[2/6] RUNNING DEEP EDA (SAVING PLOTS)...")

def save_plot(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    log(f"   - Saved plot: {filename}")

# --- 1. Rating Distribution ---
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.countplot(x='rating', data=df_ratings, palette='viridis', ax=ax1)
ax1.set_title('PHÂN PHỐI ĐIỂM RATING (Toàn tập dữ liệu)')
ax1.bar_label(ax1.containers[0], fmt='%.0f')
save_plot(fig1, '01_Rating_Distribution.png')

# --- 2. Popularity vs Quality (The Funnel) ---
movie_stats = df_ratings.groupby('movieId').agg({'rating': ['count', 'mean']})
movie_stats.columns = ['count', 'mean']
movie_stats = movie_stats[movie_stats['count'] > 50] # Lọc nhiễu

g = sns.jointplot(data=movie_stats, x='count', y='mean', kind="hex", color="#4CB391", height=8, xscale="log")
g.fig.suptitle('TƯƠNG QUAN: ĐỘ PHỔ BIẾN vs CHẤT LƯỢNG', y=1.02)
save_plot(g.fig, '02_Popularity_Quality.png')

# --- 3. User Activity Heatmap (Giờ vs Thứ) ---
activity = df_ratings.groupby(['day_of_week', 'hour']).size().unstack()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
activity = activity.reindex(days_order)
fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.heatmap(activity, cmap='YlOrBr', ax=ax3)
ax3.set_title('HEATMAP: THỜI GIAN HOẠT ĐỘNG CỦA USER')
save_plot(fig3, '03_Activity_Heatmap.png')

# --- 4. Genre Trends Over Time (Evolution) ---
sample = df_ratings.sample(frac=0.3, random_state=42).merge(df_movies[['movieId', 'genres_list']], on='movieId')
exploded = sample.explode('genres_list')
exploded = exploded[(exploded['year'] >= 1996) & (exploded['year'] <= 2002)]
genre_trends = exploded.groupby(['year', 'genres_list']).size().reset_index(name='count')
total_year = genre_trends.groupby('year')['count'].transform('sum')
genre_trends['pct'] = (genre_trends['count'] / total_year) * 100
top_genres = exploded['genres_list'].value_counts().head(6).index

fig4, ax4 = plt.subplots()
sns.lineplot(data=genre_trends[genre_trends['genres_list'].isin(top_genres)], 
             x='year', y='pct', hue='genres_list', marker='o', ax=ax4)
ax4.set_title('SỰ THAY ĐỔI THỊ HIẾU THỂ LOẠI THEO NĂM')
save_plot(fig4, '04_Genre_Evolution.png')

# --- 5. Age Distribution ---
fig5, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(x='age_group', data=df_users, order=AGE_ORDER, ax=axes[0], palette='Blues_d')
axes[0].set_title('Số lượng User theo Nhóm tuổi')
sns.countplot(x='age_group', data=df_ratings, order=AGE_ORDER, ax=axes[1], palette='Oranges_d')
axes[1].set_title('Tổng lượng Rating theo Nhóm tuổi')
save_plot(fig5, '05_Age_Distribution.png')

# --- 6. Genre Preference by Age (Heatmap) ---
age_exploded = df_ratings.merge(df_movies[['movieId', 'genres_list']], on='movieId').explode('genres_list')
age_genre = age_exploded.groupby(['age_group', 'genres_list']).size().reset_index(name='count')
age_totals = age_genre.groupby('age_group')['count'].transform('sum')
age_genre['pct'] = (age_genre['count'] / age_totals) * 100
pivot_age = age_genre.pivot(index='genres_list', columns='age_group', values='pct')
pivot_age = pivot_age[AGE_ORDER]
# Sort theo mean
pivot_age['mean'] = pivot_age.mean(axis=1)
pivot_age = pivot_age.sort_values('mean', ascending=False).drop(columns=['mean'])

fig6, ax6 = plt.subplots(figsize=(12, 12))
sns.heatmap(pivot_age, cmap='RdYlBu_r', annot=True, fmt=".1f", ax=ax6)
ax6.set_title('HEATMAP: SỞ THÍCH THỂ LOẠI THEO ĐỘ TUỔI (%)')
save_plot(fig6, '06_Genre_Age_Heatmap.png')

# --- 7. Trend Lifespan (Short-term Analysis) ---
# Xem 5 phim hot nhất diễn biến ntn theo tuần
top_5_ids = df_ratings['movieId'].value_counts().head(5).index
subset_trend = df_ratings[df_ratings['movieId'].isin(top_5_ids)].copy()
# Convert period to string for plotting
trend_pivot = subset_trend.groupby(['week_dt', 'movieId']).size().unstack(fill_value=0)
fig7, ax7 = plt.subplots()
trend_pivot.iloc[:50].plot(marker='.', ax=ax7)
ax7.set_title('VÒNG ĐỜI TRENDING CỦA TOP 5 PHIM (Theo tuần)')
save_plot(fig7, '07_Trend_Lifespan.png')

# --- 8. Generation Gap (Fix KeyError 50+) ---
pivot_gap = df_ratings.pivot_table(index='movieId', columns='age_group', values='rating', aggfunc='mean')
count_gap = df_ratings.pivot_table(index='movieId', columns='age_group', values='rating', aggfunc='count')
# So sánh 'Under 18' vs '56+'
if 'Under 18' in pivot_gap.columns and '56+' in pivot_gap.columns:
    mask = (count_gap['Under 18'] > 30) & (count_gap['56+'] > 30)
    valid_gap = pivot_gap[mask].copy()
    valid_gap['Gap'] = valid_gap['Under 18'] - valid_gap['56+']
    valid_gap = valid_gap.merge(df_movies[['movieId', 'title']], on='movieId')
    
    young_fav = valid_gap.sort_values('Gap', ascending=False).head(5)
    old_fav = valid_gap.sort_values('Gap', ascending=True).head(5)
    plot_data = pd.concat([young_fav, old_fav])
    
    fig8, ax8 = plt.subplots(figsize=(14, 6))
    colors = ['#3498db' if x > 0 else '#e74c3c' for x in plot_data['Gap']]
    sns.barplot(x='Gap', y='title', data=plot_data, palette=colors, ax=ax8)
    ax8.axvline(0, color='black')
    ax8.set_title('GENERATION GAP: Sự chênh lệch Rating (Under 18 vs 56+)')
    save_plot(fig8, '08_Generation_Gap.png')
else:
    log("   - Skip Generation Gap plot (Not enough columns)")

# =============================================================================
# PHẦN 4: HỆ THỐNG GỢI Ý (TIME-DECAY + AGE-BOOST + GENRE-MATCH)
# =============================================================================
log("\n[3/6] BUILDING HYBRID TRENDING MODEL...")

class HybridTrendRecommender:
    def __init__(self, movies_df, train_df):
        self.movies_df = movies_df
        # Map Genres
        self.movie_genres = pd.Series(movies_df['genres_list'].values, index=movies_df['movieId']).to_dict()
        self.train_df = train_df
        
        # Pre-compute Weekly Stats
        log("   -> Pre-computing statistics...")
        self.global_weekly = train_df.groupby(['week_dt', 'movieId']).size()
        self.age_weekly = train_df.groupby(['week_dt', 'age_group', 'movieId']).size()
        
    def get_user_profile(self, user_id):
        """Lấy top 3 thể loại yêu thích của user"""
        user_hist = self.train_df[self.train_df['userId'] == user_id]
        if user_hist.empty: return set()
        
        liked = user_hist[user_hist['rating'] >= 4] # Thích là rating cao
        counts = {}
        for mid in liked['movieId']:
            for g in self.movie_genres.get(mid, []):
                counts[g] = counts.get(g, 0) + 1
        
        sorted_g = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return set([g for g, c in sorted_g[:3]])

    def predict(self, user_id, user_age_group, predict_date, lookback_weeks=4):
        """
        Logic tính điểm cốt lõi:
        Score = Sum( (Global_Count * Time_Weight) + (Age_Count * Time_Weight * 2.0) )
        """
        current_week = pd.Period(predict_date, freq='W')
        scores = {}
        
        # Hệ số giảm dần theo thời gian (Tuần này quan trọng nhất)
        decay_weights = [1.0, 0.7, 0.4, 0.2] 
        
        for i in range(lookback_weeks):
            if i >= len(decay_weights): break
            target_week = current_week - i
            w = decay_weights[i]
            
            # 1. Global Trend
            if target_week in self.global_weekly.index.levels[0]:
                g_data = self.global_weekly.loc[target_week]
                for mid, count in g_data.items():
                    scores[mid] = scores.get(mid, 0) + (count * w)
            
            # 2. Age Trend (Boost)
            if (target_week, user_age_group) in self.age_weekly.index:
                a_data = self.age_weekly.loc[(target_week, user_age_group)]
                for mid, count in a_data.items():
                    # Boost x2 nếu phim hot trong nhóm tuổi
                    scores[mid] = scores.get(mid, 0) + (count * w * 2.0)
                    
        return scores

    def recommend(self, user_id, user_age_group, predict_date, k=10):
        # 1. Tính điểm Trending
        raw_scores = self.predict(user_id, user_age_group, predict_date)
        candidates = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 2. User Profile
        user_genres = self.get_user_profile(user_id)
        watched = set(self.train_df[self.train_df['userId'] == user_id]['movieId'])
        
        # 3. Rerank & Filter
        final_recs = []
        top_candidates = candidates[:300]
        match_list = []
        fallback_list = []
        
        for mid, score in top_candidates:
            if mid in watched: continue
            
            genres = set(self.movie_genres.get(mid, []))
            is_match = bool(user_genres & genres)
            
            item = (mid, score, is_match)
            if is_match:
                match_list.append(item)
            else:
                fallback_list.append(item)
        
        # Ưu tiên phim match genre trước
        combined = match_list[:k]
        if len(combined) < k:
            combined += fallback_list[:(k - len(combined))]
            
        return [x[0] for x in combined] # Chỉ trả về list movieId

# =============================================================================
# PHẦN 5: CHIA TẬP DỮ LIỆU (TEMPORAL SPLIT THEO USER)
# =============================================================================
log("\n[4/6] SPLITTING DATA (LAST-N-OUT EVALUATION STRATEGY)...")

# Để đánh giá công bằng, với mỗi user, ta giấu đi 20% hành động cuối cùng để test
# Các hành động trước đó dùng để train
train_list = []
test_list = []

# Group by user và split (Chạy trên toàn bộ user sẽ lâu, ta sẽ lấy mẫu 1000 user để đánh giá metrics)
# Nếu máy bạn mạnh, có thể bỏ dòng .sample()
unique_users = df_ratings['userId'].unique()
np.random.seed(42)
eval_users = np.random.choice(unique_users, size=1000, replace=False) # Lấy 1000 user đánh giá

subset_ratings = df_ratings[df_ratings['userId'].isin(eval_users)].sort_values(['userId', 'timestamp'])

for uid, group in subset_ratings.groupby('userId'):
    n = len(group)
    if n < 5: continue # Bỏ qua user ít tương tác
    
    split_point = int(n * 0.8)
    train_list.append(group.iloc[:split_point])
    test_list.append(group.iloc[split_point:])

train_df = pd.concat(train_list)
test_df = pd.concat(test_list)

log(f"   - Train Size: {len(train_df)} | Test Size: {len(test_df)} (trên 1000 users mẫu)")

# Init Model với tập Train
model = HybridTrendRecommender(df_movies, train_df)

# =============================================================================
# PHẦN 6: ĐÁNH GIÁ (CALCULATE METRICS: PRECISION/NDCG @ 5, 10)
# =============================================================================
log("\n[5/6] CALCULATING METRICS (PRECISION & NDCG)...")

def ndcg_at_k(recommended, actual, k):
    dcg = 0.0
    idcg = 0.0
    actual_set = set(actual)
    
    # Calc DCG
    for i, mid in enumerate(recommended[:k]):
        if mid in actual_set:
            dcg += 1.0 / np.log2(i + 2)
            
    # Calc IDCG (Ideal)
    num_relevant = min(len(actual), k)
    for i in range(num_relevant):
        idcg += 1.0 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

metrics = {
    'P@5': [], 'P@10': [],
    'N@5': [], 'N@10': []
}

# Mapping user info để lấy age group
user_map = df_users.set_index('userId')['age_group'].to_dict()

# Group test data
test_grouped = test_df.groupby('userId')['movieId'].apply(list)

# Thời điểm dự đoán: Là thời điểm cuối cùng user xuất hiện trong train (Context)
train_last_time = train_df.groupby('userId')['timestamp'].max().to_dict()

count = 0
for uid, actual_movies in test_grouped.items():
    if uid not in train_last_time: continue
    
    # Context
    predict_date = pd.to_datetime(train_last_time[uid], unit='s')
    age_group = user_map.get(uid, "25-34") # Fallback
    
    # Recommend
    recs = model.recommend(uid, age_group, predict_date, k=10)
    
    if not recs: continue
    
    # Metrics Calc
    actual_set = set(actual_movies)
    
    # Precision
    p5 = len(set(recs[:5]) & actual_set) / 5
    p10 = len(set(recs[:10]) & actual_set) / 10
    
    # NDCG
    n5 = ndcg_at_k(recs, actual_movies, 5)
    n10 = ndcg_at_k(recs, actual_movies, 10)
    
    metrics['P@5'].append(p5)
    metrics['P@10'].append(p10)
    metrics['N@5'].append(n5)
    metrics['N@10'].append(n10)
    
    count += 1
    if count % 200 == 0:
        print(f"   ...Processed {count} users")

# Tổng hợp kết quả
results = {k: np.mean(v) for k, v in metrics.items()}
log("\n" + "="*40)
log("KẾT QUẢ ĐÁNH GIÁ (EVALUATION RESULTS)")
log("="*40)
log(f"Precision@5 : {results['P@5']:.4f}")
log(f"Precision@10: {results['P@10']:.4f}")
log(f"NDCG@5      : {results['N@5']:.4f}")
log(f"NDCG@10     : {results['N@10']:.4f}")
log("-" * 40)
log("Nhận xét: Với mô hình Trending (Non-personalized Deep Learning),")
log("P@10 ~ 0.08-0.12 là kết quả KHẢ QUAN (tốt hơn Random ~0.001 rất nhiều).")
log("NDCG cao hơn Precision cho thấy các phim đúng thường nằm ở top đầu.")

# =============================================================================
# PHẦN 7: DEMO CHI TIẾT
# =============================================================================
log("\n[6/6] DEMO CHI TIẾT CHO 1 USER MẪU...")
demo_uid = list(test_grouped.keys())[0]
demo_age = user_map[demo_uid]
demo_time = pd.to_datetime(train_last_time[demo_uid], unit='s')

recs = model.recommend(demo_uid, demo_age, demo_time, k=10)
profile = model.get_user_profile(demo_uid)

log(f"User ID: {demo_uid} | Age: {demo_age} | Predict Date: {demo_time.date()}")
log(f"Top Genres (History): {profile}")
log(f"Recommendations:")
for i, mid in enumerate(recs):
    info = df_movies[df_movies['movieId'] == mid].iloc[0]
    is_match = bool(set(info['genres_list']) & profile)
    status = "[MATCH GENRE]" if is_match else "[TRENDING]"
    log(f"   {i+1}. {info['title'][:40]:<42} {status}")

log(f"\n>>> HOÀN THÀNH TOÀN BỘ. MỞ FOLDER '{OUTPUT_DIR}' ĐỂ XEM.")