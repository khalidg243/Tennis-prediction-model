from typing import Optional
from pydantic import BaseModel, Field


class MatchRecord(BaseModel):
    date: str = Field(..., description="ISO date")
    tourney_name: str
    surface: str
    round: str
    best_of: int
    winner_name: str
    loser_name: str
    w_ace: Optional[float] = None
    w_df: Optional[float] = None
    w_svpt: Optional[float] = None
    w_1stIn: Optional[float] = None
    w_1stWon: Optional[float] = None
    w_2ndWon: Optional[float] = None
    l_ace: Optional[float] = None
    l_df: Optional[float] = None
    l_svpt: Optional[float] = None
    l_1stIn: Optional[float] = None
    l_1stWon: Optional[float] = None
    l_2ndWon: Optional[float] = None


class FinalistsConfig(BaseModel):
    FINALIST_A: str
    FINALIST_B: str


