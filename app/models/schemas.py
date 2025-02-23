from typing import List, Literal, Optional
from pydantic import BaseModel

# Define possible rating values
RatingType = Literal["Needs Improvement", "Good", "Excellent", "Not implemented", "Not enough data"]

# Recommendation Schema
class Recommendation(BaseModel):
    type: Literal["Book", "Video", "Exercise"]
    title: str
    description: str
    link: Optional[str] = None 

# Technique Analysis Schema
class TechniqueAnalysis(BaseModel):
    title: str
    rating: RatingType
    feedback: Optional[List[str]] = None
    recommendations: Optional[List[Recommendation]] = None