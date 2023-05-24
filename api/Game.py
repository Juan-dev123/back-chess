from pydantic import BaseModel


class Game(BaseModel):
    id: str
    rated: bool
    created_at: str
    last_move_at: str
    turns: int
    victory_status: str
    winner: str
    increment_code: str
    white_id: str
    white_rating: int
    black_id: str
    black_rating: int
    moves: str
    opening_eco: str
    opening_name: str
    opening_ply: int
