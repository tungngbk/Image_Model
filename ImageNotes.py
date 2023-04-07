from pydantic import BaseModel


class ImageNote(BaseModel):
    image_url: str