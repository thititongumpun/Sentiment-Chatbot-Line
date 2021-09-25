from pydantic import BaseModel

class SentimentData(BaseModel):
  Sentiment: int
  SentimentText: str