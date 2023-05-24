import uvicorn
import pandas as pd
import joblib
from fastapi import FastAPI
from ListGame import ListGame
from Preprocessing import Preprocessing
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = joblib.load("best_model.joblib")

origins = [
    "https://witty-rock-08a184410.3.azurestaticapps.net/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/knn/predict')
def index(data: ListGame):
    games_ids, winners = predict_results(data)
    return {'winners': winners.tolist(),
            'games ids': games_ids}


def predict_results(games: ListGame):
    games_ids, df = convert_listgame_to_dataframe(games)
    df_processed = Preprocessing.final_model.fit_transform(df)
    print(df_processed)
    winners = model.predict(df_processed)
    return games_ids, winners


def convert_listgame_to_dataframe(games: ListGame):
    list = games.data
    values = []
    games_ids = []
    for game in list:
        row = [game.id, game.rated, game.created_at, game.last_move_at, game.turns,
               game.victory_status, game.winner, game.increment_code, game.white_id, game.white_rating,
               game.black_id, game.black_rating, game.moves, game.opening_eco, game.opening_name,
               game.opening_ply]
        values.append(row)
        games_ids.append(game.id)
    columns_names = ["id", "rated", "created_at", "last_move_at", "turns", "victory_status", "winner", "increment_code",
                     "white_id", "white_rating", "black_id", "black_rating", "moves", "opening_eco", "opening_name",
                     "opening_ply"]
    df = pd.DataFrame(data=values, columns=columns_names)
    return games_ids, df


if __name__ == '__main__':
    uvicorn.run(app)
# uvicorn main:app --reload
