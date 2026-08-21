from pydantic import BaseModel

class InferenceConfig(BaseModel):
    sequence_length: int = 30
    confidence_threshold: float = 0.85
    agreement_window: int = 10
    cooldown_seconds: float = 1.0
