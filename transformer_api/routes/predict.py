from fastapi import APIRouter
from model import predict

router = APIRouter(prefix="/predict")

@router.get("/")
def run_predict():
    return {"result": predict()}