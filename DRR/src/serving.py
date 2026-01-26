import os
import glob
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Optional, Union
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import time

# --- IMPORT MODULES CỦA BẠN ---
from recommender import DRRAgent 
from state_representation import DRRAveStateRepresentation

# --- CẤU HÌNH ---
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m')
SAVE_MODEL_DIR = "./save_model"
STATE_SIZE = 10
EMBEDDING_DIM = 100

# --- SETUP FASTAPI ---
app = FastAPI(title="DRR Recommender System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"\n❌ LỖI DỮ LIỆU (422): {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# --- GLOBAL VARIABLES ---
class GlobalState:
    agent: DRRAgent = None
    # users_dict: Dùng cho Model (chỉ chứa ID phim để tính toán)
    users_dict = None 
    # user_rich_history: Dùng cho UI (chứa đầy đủ MovieID, Rating, Timestamp)
    user_rich_history = {} 
    
    users_history_lens = None
    movies_id_to_title = {}
    users_num = 0
    items_num = 0
    all_movie_ids = None

state = GlobalState()

# --- PYDANTIC MODELS ---

# 1. Models cho Recommend
class RecommendationRequest(BaseModel):
    user_id: Union[int, str]
    top_k: int = 10

class RecItem(BaseModel):
    movie_id: int
    score: float
    title: Optional[str] = None

class RecommendationResponse(BaseModel):
    recommendations: List[RecItem]

# 2. Models cho Feedback
class FeedbackRequest(BaseModel):
    user_id: Union[int, str]
    movie_id: int
    action: str 
    rating: Optional[float] = None
    timestamp: float

class FeedbackResponse(BaseModel):
    status: str
    message: str

# 3. Models cho History (MỚI)
class HistoryResponseItem(BaseModel):
    movie_id: int
    rating: float
    timestamp: float

class HistoryResponse(BaseModel):
    history: List[HistoryResponseItem]

# --- HELPER FUNCTIONS ---
def find_latest_weights(base_dir):
    """Tìm weight mới nhất"""
    if not os.path.exists(base_dir): return None, None
    actor_files = glob.glob(os.path.join(base_dir, "**", "actor_*.weights.h5"), recursive=True)
    if not actor_files: return None, None
    actor_files.sort(key=os.path.getmtime)
    latest_actor = actor_files[-1]
    folder_path = os.path.dirname(latest_actor)
    filename = os.path.basename(latest_actor)
    match = re.search(r'actor_(\d+)', filename)
    latest_critic = None
    if match:
        ep_num = match.group(1)
        latest_critic = os.path.join(folder_path, f"critic_{ep_num}.weights.h5")
    if not latest_critic or not os.path.exists(latest_critic):
        critic_files = glob.glob(os.path.join(folder_path, "critic_*.weights.h5"))
        if critic_files:
            critic_files.sort(key=os.path.getmtime)
            latest_critic = critic_files[-1]
    return latest_actor, latest_critic

def load_data():
    """Load dữ liệu và xây dựng Rich History"""
    print(">>> Loading Data...")
    
    # 1. Load Ratings (Để tạo Rich History cho UI)
    ratings_path = os.path.join(DATA_DIR, 'ratings.dat')
    if os.path.exists(ratings_path):
        # Đọc file ratings.dat
        print("   Parsing ratings.dat for rich history...")
        ratings_list = [i.strip().split("::") for i in open(ratings_path, 'r').readlines()]
        ratings_df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        
        # Convert types
        ratings_df['UserID'] = ratings_df['UserID'].astype(int)
        ratings_df['MovieID'] = ratings_df['MovieID'].astype(int)
        ratings_df['Rating'] = ratings_df['Rating'].astype(float)
        ratings_df['Timestamp'] = ratings_df['Timestamp'].astype(float)

        state.users_num = ratings_df["UserID"].max() + 1
        state.items_num = ratings_df["MovieID"].max() + 1
        
        # Gom nhóm theo UserID để truy xuất nhanh O(1)
        # Tạo dictionary: { user_id: [ {movie_id, rating, timestamp}, ... ] }
        groups = ratings_df.groupby('UserID')
        for user_id, group in groups:
            # Chuyển DataFrame group thành list of dicts
            records = group[['MovieID', 'Rating', 'Timestamp']].to_dict('records')
            # Đổi key cho khớp với model Pydantic
            formatted_records = [
                {'movie_id': r['MovieID'], 'rating': r['Rating'], 'timestamp': r['Timestamp']}
                for r in records
            ]
            # Sort ngược theo timestamp (mới nhất lên đầu)
            formatted_records.sort(key=lambda x: x['timestamp'], reverse=True)
            state.user_rich_history[user_id] = formatted_records
            
        print(f"   Rich History built for {len(state.user_rich_history)} users.")
    else:
        print("WARNING: ratings.dat not found. History API will be empty.")
        state.users_num = 6041
        state.items_num = 3953

    # 2. Load Movies (Tiêu đề)
    movies_path = os.path.join(DATA_DIR, 'movies.dat')
    if os.path.exists(movies_path):
        movies_list = [i.strip().split("::") for i in open(movies_path, encoding='latin-1').readlines()]
        state.movies_id_to_title = {int(m[0]): m[1] for m in movies_list}

    # 3. Load User Dict (Cho Model DRR - Input State)
    try:
        state.users_dict = np.load('./data/user_dict.npy', allow_pickle=True).item()
    except Exception as e:
        print(f"WARNING: Could not load user_dict.npy: {e}")
        state.users_dict = {}

    state.all_movie_ids = np.arange(state.items_num)
    print(">>> Data Loaded Complete.")

def initialize_model():
    """Khởi tạo model"""
    state.agent = DRRAgent(env=None, users_num=state.users_num, items_num=state.items_num, 
                           state_size=STATE_SIZE, is_test=True, use_wandb=False)
    state.agent.actor.build_networks()
    state.agent.critic.build_networks()
    
    actor_path, critic_path = find_latest_weights(SAVE_MODEL_DIR)
    if actor_path and critic_path:
        print(f">>> Loading weights: {os.path.basename(actor_path)}")
        try:
            state.agent.load_model(actor_path, critic_path)
        except Exception as e:
            print(f"ERROR loading weights: {e}")

@app.on_event("startup")
async def startup_event():
    load_data()
    initialize_model()

# --- API ENDPOINTS ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "users_loaded": len(state.user_rich_history)}

# --- 1. GET HISTORY API (MỚI) ---
@app.get("/history", response_model=HistoryResponse)
async def get_history(user_id: int = Query(..., description="User ID"), limit: int = 50):
    """
    Trả về lịch sử xem phim của user, sắp xếp mới nhất -> cũ nhất.
    """
    if user_id in state.user_rich_history:
        full_hist = state.user_rich_history[user_id]
        # Lấy limit phần tử đầu tiên
        return {"history": full_hist[:limit]}
    else:
        return {"history": []}

# --- 2. RECOMMEND API ---
@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    try:
        user_id = int(request.user_id)
    except:
        user_id = 0
    
    top_k = request.top_k
    state_ids = [0] * STATE_SIZE
    
    # Lấy state từ users_dict (Model Dict)
    if state.users_dict and user_id in state.users_dict:
        history = state.users_dict[user_id]
        try:
            # history có thể là list tuple (id, rate) hoặc list id
            hist_movie_ids = [x[0] if isinstance(x, (list, tuple)) else x for x in history]
        except:
            hist_movie_ids = []
            
        if len(hist_movie_ids) < STATE_SIZE:
            state_ids = [0] * (STATE_SIZE - len(hist_movie_ids)) + hist_movie_ids
        else:
            state_ids = hist_movie_ids[-STATE_SIZE:]

    try:
        user_eb = state.agent.embedding_network.get_layer('user_embedding')(np.array(user_id))
        items_eb = state.agent.embedding_network.get_layer('movie_embedding')(np.array(state_ids))
        
        input_state = state.agent.srm_ave([
            np.expand_dims(user_eb, axis=0), 
            np.expand_dims(items_eb, axis=0)
        ])

        action = state.agent.actor.network(input_state)
        all_items_ebs = state.agent.embedding_network.get_layer('movie_embedding')(state.all_movie_ids)
        scores = tf.linalg.matmul(all_items_ebs, tf.transpose(action))
        scores = np.squeeze(scores.numpy())

        # Filter items user has seen (Dựa trên users_dict của model)
        if state.users_dict and user_id in state.users_dict:
            try:
                hist = state.users_dict[user_id]
                seen_ids = [x[0] if isinstance(x, (list, tuple)) else x for x in hist]
                scores[seen_ids] = -np.inf
            except: pass

        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            mid = int(state.all_movie_ids[idx])
            title = state.movies_id_to_title.get(mid, f"Movie ID {mid}")
            results.append({"movie_id": mid, "score": float(scores[idx]), "title": title})
            
        return {"recommendations": results}

    except Exception as e:
        print(f"Rec Error: {e}")
        return {"recommendations": []}

# --- 3. FEEDBACK API (CẬP NHẬT CẢ 2 LIST) ---
@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    # Log file
    with open("user_feedback.log", "a") as f:
        f.write(f"{request.user_id},{request.movie_id},{request.action},{request.rating},{request.timestamp}\n")
    
    u_id = int(request.user_id)
    m_id = int(request.movie_id)
    rating = request.rating if request.rating is not None else 5.0
    ts = request.timestamp

    # 1. Update Model State (users_dict) - Để lần sau Recommend chuẩn hơn
    if state.users_dict is not None:
        new_interaction_model = (m_id, rating) # Format cho model
        if u_id in state.users_dict:
            state.users_dict[u_id].append(new_interaction_model)
            if len(state.users_dict[u_id]) > 100:
                 state.users_dict[u_id] = state.users_dict[u_id][-100:]
        else:
            state.users_dict[u_id] = [new_interaction_model]

    # 2. Update UI History (user_rich_history) - Để hiển thị ngay lên Sidebar
    new_interaction_ui = {'movie_id': m_id, 'rating': rating, 'timestamp': ts}
    
    if u_id in state.user_rich_history:
        # Chèn vào đầu list (vì sort mới nhất ở đầu)
        state.user_rich_history[u_id].insert(0, new_interaction_ui)
    else:
        state.user_rich_history[u_id] = [new_interaction_ui]

    print(f"Updated State & History for User {u_id}")
    return {"status": "success", "message": "Logged & Memory Updated"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)