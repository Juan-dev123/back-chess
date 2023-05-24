from pydantic import BaseModel
from typing import List
from Game import Game


class ListGame(BaseModel):
    data: List[Game]