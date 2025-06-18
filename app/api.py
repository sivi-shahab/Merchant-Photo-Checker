import asyncpg
import asyncio
from fastapi import APIRouter, HTTPException, Query, FastAPI
from pymongo import MongoClient
from app.processor import ImageProcessor
from app.utils import decode_image_base64_to_tempfile
from app.config import settings
from app.logger import logger
import os
import requests
import time


router = APIRouter()
# processor = ImageProcessor()

# Setup Mongo client
mongo_client = MongoClient(
    settings.MONGO_URI,
    serverSelectionTimeoutMS=60000,  # 60 seconds
    connectTimeoutMS=30000,          # 30 seconds  
    socketTimeoutMS=30000,           # 30 seconds
    maxPoolSize=10,                  # Connection pool size
    retryWrites=True                 # Enable retry writes
)
mongo_db = mongo_client[settings.MONGO_DB]
mongo_collection = mongo_db[settings.MONGO_COLLECTION]

# Record start time for uptime
start_time = time.time()
# Global pool dan processor
postgres_pool = None
processor = None

async def setup_resources():
    global postgres_pool, processor
    if postgres_pool is None:
        postgres_pool = await asyncpg.create_pool(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            database=settings.POSTGRES_DB,
            min_size=1,
            max_size=5
        )
    if processor is None:
        processor = ImageProcessor(mongo_collection, postgres_pool=postgres_pool)

def register_events(app: FastAPI):
    @app.on_event("startup")
    async def on_startup():
        await setup_resources()

@router.get("/health")
async def health_check():
    # MongoDB check
    try:
        mongo_db.command("ping")
        mongo_status = "ok"
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        mongo_status = f"error: {str(e)}"

    # Ollama check
    try:
        response = requests.get("http://ollama:11434/api/tags", timeout=5)
        if response.status_code == 200:
            ollama_status = "ok"
        else:
            ollama_status = f"error: status {response.status_code}"
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        ollama_status = f"error: {str(e)}"

    uptime_seconds = int(time.time() - start_time)

    return {
        "status": "healthy" if mongo_status == "ok" and ollama_status == "ok" else "degraded",
        "mongo_status": mongo_status,
        "ollama_status": ollama_status,
        "uptime_seconds": uptime_seconds
    }
    
@router.get("/ocr/{reffid}")
async def ocr_images_by_reffid(reffid: str):
    """
    Fetches all validated images (base64) for a given reffid,
    runs OCR via Ollama Llama3.5-Vision, updates MongoDB & Postgres,
    and returns the OCR summary.
    """
    try:
        # 1. Load the document
        doc = processor.get_document_by_reffid(reffid)
        if not doc:
            raise HTTPException(status_code=404, detail="Reffid not found in MongoDB")

        images = doc.get("images", [])
        # Filter only validated images
        valid_images = [
            img for img in images
            if img.get("validated") and img.get("image_base64")
        ]

        # 2. Process each image in parallel
        async def handle_image(image_data):
            elk_id = image_data["elk_id"]
            base64_str = image_data["image_base64"]
            # ensure the "data:image" prefix
            if not base64_str.startswith("data:image"):
                base64_str = f"data:image/jpeg;base64,{base64_str}"

            try:
                ocr = await processor.process_image_with_ollama_chat(base64_str)
                # update MongoDB
                upd = processor.mongo_collection.update_one(
                    {"reffid": reffid, "images.elk_id": elk_id},
                    {"$set": {"images.$.ocr_result": ocr.get("result", "")}}
                )
                return {
                    "elk_id": elk_id,
                    "ocr_result": ocr.get("result", ""),
                    "updated_in_mongo": upd.modified_count > 0
                }

            except Exception as err:
                logger.error(f"[reffid={reffid} elk_id={elk_id}] OCR error: {err}")
                return {
                    "elk_id": elk_id,
                    "error": "Failed to process image",
                    "details": str(err)
                }

        ocr_results = await asyncio.gather(*[handle_image(img) for img in valid_images])

        # 3. Push top-3 store names into Postgres
        await processor.update_ocr_store_names(processor.postgres_pool, reffid, ocr_results)

        # 4. Return summary
        return {
            "reffid": reffid,
            "total_images": len(images),
            "validated_images": len(valid_images),
            "processed_images": len(ocr_results),
            "ocr_results": ocr_results
        }

    except HTTPException:
        # re-raise 404
        raise
    except Exception as e:
        logger.error(f"OCR endpoint error for reffid={reffid}: {e}")
        raise HTTPException(status_code=500, detail="Internal OCR processing error")


# @router.get("/ocr/{reffid}")
# async def ocr_images_by_reffid(reffid: str):
#     """
#     Process OCR for all validated images associated with a given reffid.
#     Returns OCR results and updates MongoDB with the results.
#     """
#     try:
#         # Get document from MongoDB
#         doc = processor.get_document_by_reffid(reffid)
#         if not doc:
#             raise HTTPException(status_code=404, detail="Reffid not found in MongoDB")

#         images = doc.get("images", [])
#         ocr_results = []

#         for image_data in images:
#             if not image_data.get("validated"):
#                 continue

#             try:
#                 # Process image directly from base64 without temp file
#                 base64_str = image_data["image_base64"]
#                 if not base64_str.startswith('data:image'):
#                     base64_str = f"data:image/jpeg;base64,{base64_str}"

#                 # ocr_result = await processor.process_image_with_ollama(base64_str)
#                 ocr_result = await processor.process_image_with_ollama_chat(base64_str)

#                 # Update MongoDB with OCR result
#                 update_result = processor.mongo_collection.update_one(
#                     {"reffid": reffid, "images.elk_id": image_data["elk_id"]},
#                     {"$set": {"images.$.ocr_result": ocr_result.get("result", "")}}
#                 )

#                 ocr_results.append({
#                     "elk_id": image_data["elk_id"],
#                     "ocr_result": ocr_result.get("result", ""),
#                     "updated": update_result.modified_count > 0
#                 })

#             except Exception as img_error:
#                 logger.error(f"Error processing image {image_data.get('elk_id')}: {str(img_error)}")
#                 ocr_results.append({
#                     "elk_id": image_data["elk_id"],
#                     "error": "Failed to process image",
#                     "details": str(img_error)
#                 })
                
#         await processor.update_ocr_store_names(processor.postgres_pool, reffid, ocr_results)
#         return {
#             "reffid": reffid,
#             "total_images": len(images),
#             "processed_images": len(ocr_results),
#             "ocr_results": ocr_results
#         }

#     except Exception as e:
#         logger.error(f"Error in OCR endpoint for reffid {reffid}: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))



@router.get("/detect-blur/{reffid}")
async def detect_blur_images_by_reffid(reffid: str):
    doc = processor.get_document_by_reffid(reffid)
    if not doc:
        raise HTTPException(status_code=404, detail="Reffid not found in MongoDB")

    images = doc.get("images", [])
    blur_results = []

    for image_data in images:
        if image_data.get("validated"):
            temp_file = decode_image_base64_to_tempfile(image_data["image_base64"])
            blur_result = processor.detect_image_blur(temp_file)

            # Optional: update MongoDB
            processor.mongo_collection.update_one(
                {"reffid": reffid, "images.elk_id": image_data["elk_id"]},
                {"$set": {"images.$.blur_result": blur_result}}
            )

            blur_results.append({
                "elk_id": image_data["elk_id"],
                "blur_result": blur_result
            })

            os.unlink(temp_file)
            
    await processor.update_blur_scores(processor.postgres_pool, reffid, blur_results)

    return {
        "reffid": reffid,
        "total_images": len(blur_results),
        "blur_results": blur_results
    }
    
@router.get("/batch-ocr")
async def batch_ocr_by_created_date(start_date: str = Query(...), end_date: str = Query(...)):
    docs = processor.mongo_collection.find({
        "created_date": {"$gte": start_date, "$lte": end_date}
    })

    batch_results = []
    for doc in docs:
        reffid = doc["reffid"]
        images = doc.get("images", [])
        ocr_results = []

        for image_data in images:
            if image_data.get("validated"):
                temp_file = decode_image_base64_to_tempfile(image_data["image_base64"])
                with open(temp_file, "rb") as f:
                    img_bytes = f.read()

                ocr_result = await processor.process_image_with_ollama(img_bytes)

                # Update MongoDB
                processor.mongo_collection.update_one(
                    {"reffid": reffid, "images.elk_id": image_data["elk_id"]},
                    {"$set": {"images.$.ocr_result": ocr_result["result"]}}
                )

                ocr_results.append({
                    "elk_id": image_data["elk_id"],
                    "ocr_result": ocr_result["result"]
                })

                os.unlink(temp_file)

        batch_results.append({
            "reffid": reffid,
            "total_images": len(ocr_results),
            "ocr_results": ocr_results
        })

    return {
        "filter": {
            "start_date": start_date,
            "end_date": end_date
        },
        "total_reffids": len(batch_results),
        "batch_ocr_results": batch_results
    }

@router.get("/batch-detect-blur")
async def batch_detect_blur_by_created_date(start_date: str = Query(...), end_date: str = Query(...)):
    docs = processor.mongo_collection.find({
        "created_date": {"$gte": start_date, "$lte": end_date}
    })

    batch_results = []
    for doc in docs:
        reffid = doc["reffid"]
        images = doc.get("images", [])
        blur_results = []

        for image_data in images:
            if image_data.get("validated"):
                temp_file = decode_image_base64_to_tempfile(image_data["image_base64"])
                blur_result = processor.detect_image_blur(temp_file)

                # Update MongoDB
                processor.mongo_collection.update_one(
                    {"reffid": reffid, "images.elk_id": image_data["elk_id"]},
                    {"$set": {"images.$.blur_result": blur_result}}
                )

                blur_results.append({
                    "elk_id": image_data["elk_id"],
                    "blur_result": blur_result
                })

                os.unlink(temp_file)

        batch_results.append({
            "reffid": reffid,
            "total_images": len(blur_results),
            "blur_results": blur_results
        })

    return {
        "filter": {
            "start_date": start_date,
            "end_date": end_date
        },
        "total_reffids": len(batch_results),
        "batch_blur_results": batch_results
    }
