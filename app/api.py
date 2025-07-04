import asyncpg
import asyncio
from fastapi import APIRouter, HTTPException, Query, FastAPI,  UploadFile, File
from pymongo import MongoClient
from app.processor import ImageProcessor
from app.utils import decode_image_base64_to_tempfile
from app.config import settings
from app.logger import logger
import os
import requests
import time
import json

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
        valid_images = []
        for img in images:
            b64 = img.get("image_base64")
            if isinstance(b64, str) and b64.strip():
                valid_images.append(img)

        # Jika tidak ada gambar valid, langsung return summary
        if not valid_images:
            print("Tidak ada image yang valid")
            return {
                "reffid": reffid,
                "total_images": len(images),
                "validated_images": 0,
                "processed_images": 0,
                "ocr_results": []
            }

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

    
@router.get("/ocr_merchant/{reffid}")
async def ocr_images_by_reffid(reffid: str):
    """
    Process OCR for all validated images associated with a given reffid.
    Returns OCR results and updates MongoDB + Postgres with the results.
    """
    try:
        # Fetch document
        doc = processor.get_document_by_reffid(reffid)
        if not doc:
            raise HTTPException(status_code=404, detail="Reffid not found in MongoDB")

        images = doc.get("images", [])
        ocr_results = []

        for image_data in images:
            if not image_data.get("validated"):
                continue

            try:
                # Get raw base64 (may include prefix)
                b64 = image_data.get("image_base64", "")

                # Call updated processor (strips prefix internally)
                ocr = processor.perform_ocr_from_base64(b64)

                ocr_data = json.loads(ocr) if ocr else {}

                # Update MongoDB
                upd = processor.mongo_collection.update_one(
                    {"reffid": reffid, "images.elk_id": image_data.get("elk_id")},
                    {"$set": {"images.$.ocr_result": ocr_data.get("result", "")} }
                )

                ocr_results.append({
                    "elk_id": image_data.get("elk_id"),
                    "ocr_result": ocr_data.get("result", ""),
                    "updated": upd.modified_count > 0
                })

            except Exception as img_err:
                logger.error(f"Error processing image {image_data.get('elk_id')}: {img_err}")
                ocr_results.append({
                    "elk_id": image_data.get("elk_id"),
                    "error": "Failed to process image",
                    "details": str(img_err)
                })

        # Persist to Postgres
        await processor.update_ocr_store_names(processor.postgres_pool, reffid, ocr_results)

        return {
            "reffid": reffid,
            "total_images": len(images),
            "processed_images": len(ocr_results),
            "ocr_results": ocr_results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in OCR endpoint for reffid {reffid}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



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
    
@router.post("/ocr", summary="Perform OCR on an uploaded image")
async def ocr_image(file: UploadFile = File(...)):
    # Validasi tipe file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image.")
    try:
        result = await processor.process_image_with_ollama_chat_image(file)
        return result
    except Exception as e:
        logger.error(f"OCR endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal OCR processing error")

@router.post("/ocr_llava", summary="OCR dari file gambar")
async def ocr_upload(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Harap upload file gambar")
    return await processor.process_image_with_ollama_from_file(file)


@router.post("/llava", summary="OCR via LLava (ollama-python)")
async def ocr_llava(file: UploadFile = File(...)):
    # 1. Validasi tipe file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Harap upload file gambar saja.")
    # 2. Proses dengan LLavaProcessor
    result = await processor.process_image_llava(file)
    # 3. Jika error, konversi ke HTTPException
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["details"])
    return result


@router.get("/batch-ocr-merchant")
async def batch_ocr_by_created_date(
    start_date: str = Query(..., description="ISO date-time (YYYY-MM-DD)"),
    end_date:   str = Query(..., description="ISO date-time (YYYY-MM-DD)")
):
    """
    Run OCR for every validated image in documents created between
    `start_date` and `end_date` (inclusive).  
    Results are written back to MongoDB **and** Postgres.
    """
    try:
        # Ambil semua dokumen dalam rentang tanggal.
        docs = processor.mongo_collection.find({
            "created_date": {"$gte": start_date, "$lte": end_date}
        })

        batch_results = []

        async for doc in docs:         # ← jika .find() di-wrap motor/async
            reffid  = doc["reffid"]
            images  = doc.get("images", [])
            ocr_results = []

            for image_data in images:
                # Lewati gambar yg belum tervalidasi
                if not image_data.get("validated"):
                    continue

                try:
                    # Base64 sudah cukup – strip prefix dikerjakan di processor
                    b64 = image_data.get("image_base64", "")

                    ocr = await processor.process_image_with_ollama(b64)

                    # Update Mongo
                    upd = processor.mongo_collection.update_one(
                        {"reffid": reffid, "images.elk_id": image_data.get("elk_id")},
                        {"$set": {"images.$.ocr_result": ocr.get("result", "")}}
                    )

                    ocr_results.append({
                        "elk_id":   image_data.get("elk_id"),
                        "ocr_result": ocr.get("result", ""),
                        "updated":   upd.modified_count > 0
                    })

                except Exception as img_err:
                    logger.error(f"[batch-ocr] {reffid} img {image_data.get('elk_id')} → {img_err}")
                    ocr_results.append({
                        "elk_id": image_data.get("elk_id"),
                        "error":  "Failed to process image",
                        "details": str(img_err)
                    })

            # Simpan hasil ke Postgres (nama toko per gambar)
            await processor.update_ocr_store_names(
                processor.postgres_pool, reffid, ocr_results
            )

            batch_results.append({
                "reffid":            reffid,
                "total_images":      len(images),
                "processed_images":  len(ocr_results),
                "ocr_results":       ocr_results
            })

        return {
            "filter": {
                "start_date": start_date,
                "end_date":   end_date
            },
            "total_reffids":       len(batch_results),
            "batch_ocr_results":   batch_results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[batch-ocr] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@router.get("/merchant-tbfu")
async def get_filtered_company_profiles():
    """
    Mengambil data company profile dengan kondisi:
    - Semua kolom OCR store name (1, 2, 3) adalah NULL
    - Salah satu dari blur score (1, 2, 3) berisi 'poor'
    """
    global postgres_pool
    
    # Ensure resources are set up
    if postgres_pool is None:
        await setup_resources()
    
    try:
        # Use the global pool instead of creating new connections
        async with postgres_pool.acquire() as conn:
            query = """
            SELECT 
                mis_date, 
                reffid, 
                created_date, 
                merchant_name, 
                mid, 
                ocr_store_name_1, 
                ocr_store_name_2, 
                ocr_store_name_3, 
                blur_score_1, 
                blur_score_2, 
                blur_score_3, 
                ocr_processed_date, 
                blur_processed_date
            FROM public.mpos_company_profile
            WHERE
            (
                (CASE WHEN ocr_store_name_1 IS NULL THEN 1 ELSE 0 END) +
                (CASE WHEN ocr_store_name_2 IS NULL THEN 1 ELSE 0 END) +
                (CASE WHEN ocr_store_name_3 IS NULL THEN 1 ELSE 0 END)
            ) >= 2
            AND (
                (CASE WHEN blur_score_1 IN ('poor', 'very poor') THEN 1 ELSE 0 END) +
                (CASE WHEN blur_score_2 IN ('poor', 'very poor') THEN 1 ELSE 0 END) +
                (CASE WHEN blur_score_3 IN ('poor', 'very poor') THEN 1 ELSE 0 END)
            ) >= 2
        ORDER BY created_date DESC;
            """
            
            rows = await conn.fetch(query)
            
            if not rows:
                return []
            
            # Convert rows to list of dictionaries
            result = [dict(row) for row in rows]
            return result
            
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")