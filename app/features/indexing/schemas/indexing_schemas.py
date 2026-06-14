from pydantic import BaseModel
from typing import List


class IndexSyncRequest(BaseModel):
    product_uuids: List[str]
