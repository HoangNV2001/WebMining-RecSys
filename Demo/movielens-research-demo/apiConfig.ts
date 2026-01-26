/**
 * API CONFIGURATION & DOCUMENTATION
 * 
 * This file defines the contract between the Frontend and the ML Backend.
 * Implement your Python/FastAPI backend to match these specifications.
 */

// Change this to your actual backend URL when ready
export const API_BASE_URL = "http://localhost:8002";

export const API_ENDPOINTS = {
  HEALTH: `${API_BASE_URL}/health`, // Endpoint to check connection
  RECOMMEND: `${API_BASE_URL}/recommend`,
  FEEDBACK: `${API_BASE_URL}/feedback`,
  HISTORY: `${API_BASE_URL}/history`, // NEW: Fetch user history
};

export const API_CONFIG = {
  BASE_URL: API_BASE_URL,
  ENDPOINTS: API_ENDPOINTS,
};

// --- REQUEST/RESPONSE TYPES ---

/**
 * GET /health
 * Returns: {"status": "ok"}
 */

/**
 * POST /recommend
 * Expected Backend Logic:
 * 1. Receive user_id
 * 2. Retrieve user vector/history
 * 3. Return list of top_k movie IDs with scores
 */
export interface RecommendationRequest {
  user_id: number;
  top_k: number;
}

export interface RecommendationResponseItem {
  movie_id: number;
  score: number; // Probability or Similarity score
}

export interface RecommendationResponse {
  recommendations: RecommendationResponseItem[];
}

/**
 * POST /feedback
 * Expected Backend Logic:
 * 1. Log the interaction for online learning or offline retraining
 * 2. Update immediate session context if using RNN/Transformer models
 */
export interface FeedbackRequest {
  user_id: number;
  movie_id: number;
  action: 'like' | 'dislike' | 'rate' | 'click';
  rating?: number; // Optional, required if action is 'rate'
  timestamp: number; // Unix timestamp
}

export interface FeedbackResponse {
  status: 'success' | 'error';
  message?: string;
}

/**
 * GET /history
 * Query Params: ?user_id=1&limit=50
 * Expected Backend Logic:
 * 1. Return the most recent ratings/interactions for this user from the DB
 */
export interface HistoryResponseItem {
  movie_id: number;
  rating: number;
  timestamp: number;
}

export interface HistoryResponse {
  history: HistoryResponseItem[];
}

/**
 * EXAMPLE CURL COMMANDS FOR YOUR BACKEND:
 * 
 * 1. Check Health:
 * curl http://localhost:8000/health
 * 
 * 2. Get Recommendations:
 * curl -X POST http://localhost:8000/recommend \
 *      -H "Content-Type: application/json" \
 *      -d '{"user_id": 1, "top_k": 10}'
 * 
 * 3. Send Feedback:
 * curl -X POST http://localhost:8000/feedback \
 *      -H "Content-Type: application/json" \
 *      -d '{"user_id": 1, "movie_id": 1193, "action": "rate", "rating": 5, "timestamp": 123456789}'
 * 
 * 4. Get History:
 * curl "http://localhost:8000/history?user_id=1&limit=20"
 */