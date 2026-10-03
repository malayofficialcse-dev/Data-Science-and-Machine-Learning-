from pydantic import BaseModel

class HousePredictionRequest(BaseModel):
    area:float
    bedrooms:int
    bathrooms:int
    age:float