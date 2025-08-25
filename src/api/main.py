from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import torch
import numpy as np
import berserk
import os


from lichess_data_loading.gm_usernames import Legal_names
from data.parse_pgn import pgn_to_3d_arrays, transform_board_arrays_to_samples

app = FastAPI(title="Chess Similarity API")

class PredictionRequest(BaseModel):
    username: str
    max_games: int = 10

class PredictionResponse(BaseModel):
    username: str
    most_similar_player: str
    similarity_scores: dict[str, float]

# Global variables
model = None
lichess_client = None

@app.on_event("startup")
async def startup_event():
    global model, lichess_client
    
    # Load model
    try:
        model = mlflow.pytorch.load_model("models:/current_best/latest")
        model.eval()
    except:
        try:
            model = mlflow.pytorch.load_model("models:/current_best/1")
            model.eval()
        except:
            model = None
    
    # Load Lichess client
    try:
        token_path = os.path.join(os.path.dirname(__file__), '..', 'lichess_data_loading', 'api_token.txt')
        with open(token_path, 'r') as f:
            token = f.read().strip()
        session = berserk.TokenSession(token)
        lichess_client = berserk.Client(session)
    except:
        lichess_client = None

def download_games(username: str, max_games: int = 10) -> list[str]:
    if lichess_client is None:
        raise HTTPException(status_code=500, detail="Lichess client not available")
    
    games = []
    try:
        for game_pgn in lichess_client.games.export_by_player(
            username=username, as_pgn=True, max=max_games, rated=True
        ):
            if game_pgn:
                games.append(game_pgn)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading games: {str(e)}")
    
    if not games:
        raise HTTPException(status_code=404, detail=f"No games found for user {username}")
    
    return games

def process_games(username: str, games: list[str]) -> torch.Tensor:
    all_samples = []
    
    for pgn in games:
        try:
            lines = pgn.split('\n')
            white_player = black_player = None
            
            for line in lines:
                if line.startswith('[White "'):
                    white_player = line.split('"')[1]
                elif line.startswith('[Black "'):
                    black_player = line.split('"')[1]
            
            # Determine user color
            if white_player and white_player.lower() == username.lower():
                user_color = "white"
            elif black_player and black_player.lower() == username.lower():
                user_color = "black"
            else:
                continue
            
            board_states = pgn_to_3d_arrays(pgn)
            samples = transform_board_arrays_to_samples(user_color, board_states)
            
            if samples:
                all_samples.append(torch.tensor(np.array(samples), dtype=torch.float32))
        except:
            continue
    
    if not all_samples:
        raise HTTPException(status_code=400, detail="No valid games could be processed")
    
    return torch.cat(all_samples, dim=0)

def predict_similarity(features: torch.Tensor) -> dict[str, float]:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    with torch.no_grad():
        batch_size = features.shape[0]
        model_input = features.view(batch_size, -1, 8, 8)
        outputs = model(model_input)
        probabilities = torch.softmax(outputs, dim=1)
        avg_probabilities = torch.mean(probabilities, dim=0)
        
        return {player: float(avg_probabilities[i]) for i, player in enumerate(Legal_names)}

@app.post("/predict", response_model=PredictionResponse)
async def predict_player_similarity(request: PredictionRequest):
    games = download_games(request.username, request.max_games)
    features = process_games(request.username, games)
    similarity_scores = predict_similarity(features)
    most_similar_player = max(similarity_scores, key=similarity_scores.get)
    
    return PredictionResponse(
        username=request.username,
        most_similar_player=most_similar_player,
        similarity_scores=similarity_scores
    )
