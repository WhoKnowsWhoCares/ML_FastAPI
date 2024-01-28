from pydantic import BaseModel
from joblib import load


class Model(BaseModel):
    """ML model class"""


def load_model(path: str) -> Model:
    try:
        model = load(path)
    except FileNotFoundError as e:
        print(f'No such model: {e}')
        return None
    return model
