from fastapi import APIRouter, UploadFile, HTTPException



test_router = APIRouter()


@test_router.get("/test")
async def testing():

    return {"testing": "testing"}
