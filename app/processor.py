import os
import re
import cv2
import time
import base64
import tempfile
import numpy as np
import ollama
#import asyncpg
#from io import BytesIO
from fastapi import UploadFile
from pymongo import MongoClient
from app.config import settings
from app.logger import logger
import datetime
# import logging
import json
import requests

# logger = logging.getLogger(__name__)

# Initialize MongoDB client (tidak berubah)
mongo_client = MongoClient(settings.MONGO_URI)
mongo_db = mongo_client[settings.MONGO_DB]
mongo_collection = mongo_db[settings.MONGO_COLLECTION]

SYSTEM_PROMPT = ("""
You are a highly accurate OCR system specialized in extracting business names from images.
Follow these exact steps:
1. Analyze the image for any business name storefront, including shops, restaurants, food stalls, workshops, or service-based businesses displayed on buildings or banners.
2. If no business name storefront is found, {"return" : "null"} example id card.
3. Identify any text that represents a business name, such as shop, restaurant, food stall, workshop, or service business names on buildings or banners.
4. Select the single most likely business name based on size, prominence, and location.
**Do not description, analysis, commentary, reasoning, or hallucination— just return the result in the json format.**
Output Format in json:
{"result" : "<store name>"}
{"result" : "null"}
temperature=0.0

Prioritize / Focus On:
✓ Storefront signs
✓ Business banners
✓ Shop names on buildings
✓ Restaurant names
✓ Food‑stall names
✓ Service‑business names

Ignore:
✗ Product names
✗ Prices
✗ Addresses
✗ Phone numbers
✗ General text or descriptions
""")


class ImageProcessor:
    def __init__(self, mongo_collection=mongo_collection, postgres_pool=None):
        self.mongo_collection = mongo_collection
        self.start_time = time.time()
        self.ollama_client = ollama.Client(host=str(settings.OLLAMA_BASE_URL))
        self.postgres_pool = postgres_pool  # ← Pool Postgres

    def get_document_by_reffid(self, reffid: str) -> dict:
        """
        Retrieve the MongoDB document for a given reffid.
        """
        return self.mongo_collection.find_one({"reffid": reffid})

    def parse_response(response_text: str) -> str:
        """
        Gabungkan konten streaming dari Ollama, lalu cari blok JSON
        { "result" : "<nama>" } — return hanya bagian itu.

        Jika hasil sudah persis JSON, kembalikan langsung tanpa modifikasi.
        """
        try:
            combined_result = []

            for line in response_text.splitlines():
                try:
                    parsed = json.loads(line)
                    content = parsed.get("message", {}).get("content", "")
                    if content:
                        combined_result.append(content)  # tidak .strip()
                except json.JSONDecodeError:
                    combined_result.append(line)

            full_text = "".join(combined_result)

            # 1) Jika hasil sudah bersih (hanya JSON), langsung return
            full_json_re = r'^\s*\{\s*"result"\s*:\s*"[^"]*"\s*\}\s*$'
            if re.match(full_json_re, full_text, flags=re.IGNORECASE):
                return full_text.strip()

            # 2) Jika ada campuran teks + JSON, ekstrak bagian JSON saja
            snippet_re = r'\{\s*["\']result["\']\s*:\s*["\']([^"\']+)["\']\s*\}'
            match = re.search(snippet_re, full_text)
            if match:
                value = match.group(1)
                return f'{{ "result": "{value}" }}'

            # 3) Tidak ditemukan
            return None
        except Exception as e:
            print("⚠️ parse_response error:", str(e))
            return None

    def perform_ocr_from_base64(self, base64_image: str) -> str | None:
        """Perform OCR and extract only the JSON result string."""
        try:
            response = requests.post(
                "http://ollama:11434/api/chat",
                json={
                    "model": "llama3.2-vision",
                    "messages": [
                        {
                            "role": "user",
                            "content": SYSTEM_PROMPT,
                            "images": [base64_image],
                        },
                    ],
                }
            )

            if response.status_code != 200:
                print(f"❌ Error: {response.status_code} - {response.text}")
                return None

            # Gabungkan hasil streaming dari Ollama
            combined_result = []
            for line in response.text.splitlines():
                try:
                    parsed = json.loads(line)
                    content = parsed.get("message", {}).get("content", "")
                    if content:
                        combined_result.append(content)
                except json.JSONDecodeError:
                    combined_result.append(line)

            full_text = "".join(combined_result)

            # Jika hasil sudah bersih
            full_json_re = r'^\s*\{\s*"result"\s*:\s*"[^"]*"\s*\}\s*$'
            if re.match(full_json_re, full_text, flags=re.IGNORECASE):
                return full_text.strip()

            # Jika masih ada narasi, ambil JSON-nya saja
            snippet_re = r'\{\s*["\']result["\']\s*:\s*["\']([^"\']+)["\']\s*\}'
            match = re.search(snippet_re, full_text)
            if match:
                value = match.group(1)
                return f'{{ "result": "{value}" }}'

            return None

        except Exception as e:
            print("⚠️ Exception:", str(e))
            return None

    async def process_image_with_ollama_chat(self, image_base64: str) -> dict:
        # Debugging logs
        logger.debug(f"Ollama URL: {settings.OLLAMA_BASE_URL}")
        logger.debug(f"Ollama model: {settings.OLLAMA_MODEL}")
        try:
            if image_base64.startswith('data:image'):
                base64_data = image_base64.split(',', 1)[1]
            else:
                base64_data = image_base64

            # 2. Decode & write to temp file
            img_bytes = base64.b64decode(base64_data)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
                
            logger.debug("Wrote temp file: %s", tmp_path)
            prompt = ("""
                Analyze the image and extract store information following these steps:  
                1. Identify all banners/signage/billboards in the image.  
                2. Detect and select the largest/most prominent banner or signage or billboard.  
                3. Perform OCR on the selected largest banner/signage/billboard.
                4. Ignore all advertisements, promotional text, product brand in banner/signage/billboard.
                5. Extract ONLY the store name from the banner/signage/billboard.
                6. Return the result in exactly this format (do not modify the format):
                {result: "STORE_NAME"}
                
                Examples:
                For a store "BUDI JAYA":
                {result: "BUDI JAYA"}
                
                For a store "TOKO BERKAH":
                {result: "TOKO BERKAH"}
                
                If no store name is found:
                {result: "null"}
                
                Important:
                - Return ONLY the exact store name as seen on the banner/signage
                - Do not include any steps, analysis, or additional text
                - The response should contain ONLY the {result: "STORE_NAME"} format
                """)
            
            logger.debug("Using prompt: %s", prompt.strip().replace("\n", " "))

            # 4. Call module-level API (supports images=[])
            response = self.ollama_client.chat(
                model=settings.OLLAMA_MODEL,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [tmp_path]
                }],
            )
            try:
                os.unlink(tmp_path)
                logger.debug("Deleted temp file: %s", tmp_path)
            except Exception as rm_err:
                logger.warning("Failed to delete temp file %s: %s", tmp_path, rm_err)
            # 6. Parse the first choice’s content
            choices = response.get('choices', [])
            logger.debug("Parsed choices: %s", choices)
            content = ""
            if choices:
                content = choices[0].get('message', {}).get('content', "")
            else:
                # fallback if the API returns a top-level 'response'
                content = response.get('response', "")
            logger.debug("Extracted content: %r", content)
            store_name = self.extract_store_name(content)
            logger.debug("Parsed store_name: %r", store_name)


            return {
                "result": store_name,
                "raw_response": None if store_name != "null" else content
            }

        except Exception as e:
            logger.error(f"Error in process_image_with_ollama_chat: {str(e)}")
            return {
                "error": "OCR processing failed",
                "details": str(e)
            }

    async def process_image_with_ollama_chat_image(self, image_file: UploadFile) -> dict:
        # Debugging logs
        logger.debug(f"Ollama URL: {settings.OLLAMA_BASE_URL}")
        logger.debug(f"Ollama model: {settings.OLLAMA_MODEL}")

        try:
            # 1. Baca langsung bytes dari UploadFile
            img_bytes = await image_file.read()

            # 2. Tulis ke temporary file
            suffix = os.path.splitext(image_file.filename)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
            logger.debug("Wrote temp file: %s", tmp_path)

            # 3. Prompt tetap sama (tidak diubah)
            prompt = """
                    Analyze the image and extract store information following these steps:  
                    1. Identify all banners/signage/billboards in the image.  
                    2. Detect and select the largest/most prominent banner or signage or billboard.  
                    3. Perform OCR on the selected largest banner/signage/billboard.
                    4. Ignore all advertisements, promotional text, product brand in banner/signage/billboard.
                    5. Extract ONLY the store name from the banner/signage/billboard.
                    6. Return the result in exactly this format (do not modify the format):
                    {result: "STORE_NAME"}

                    If no store name is found:
                    {result: "null"}

                    Important:
                    - Return ONLY the exact store name as seen on the banner/signage
                    - Do not include any steps, analysis, or additional text
                    - The response should contain ONLY the {result: "STORE_NAME"} format
    
                """

            # 4. Panggil Ollama dengan images=[tmp_path]
            response = self.ollama_client.chat(
                model=settings.OLLAMA_MODEL,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [tmp_path]
                }],
            )

            # 5. Hapus temporary file segera setelah dipakai
            os.unlink(tmp_path)

            # 6. Parse response
            choices = response.get('choices', [])
            content = ""
            if choices:
                content = choices[0].get('message', {}).get('content', "")
            else:
                content = response.get('response', "")

            store_name = self.extract_store_name(content)
            return {
                "result": store_name
            }

        except Exception as e:
            logger.error(f"Error in process_image_with_ollama_chat: {e}")
            return {
                "error": "OCR processing failed",
                "details": str(e)
            }
            
    

    # async def process_image_with_ollama(self, image_base64: str) -> dict:
    #     try:
    #         # Strip data URI prefix if present
    #         if image_base64.startswith('data:image'):
    #             base64_data = image_base64.split(',', 1)[1]
    #         else:
    #             base64_data = image_base64
    #         # Decode to bytes and wrap in BytesIO
    #         decoded = base64.b64decode(base64_data)
    #         # image_io = BytesIO(decoded)
    #         print("base64 dari mongo")
    #         print(decoded)
    #
    #         # Call Ollama with in-memory image
    #         response = self.ollama_client.generate(
    #             model="llava",
    #             prompt="""
    #                 Analyze the image and extract store information following these steps:
    #                 1. Identify all banners/signage/billboards in the image.
    #                 2. Detect and select the largest/most prominent banner or signage or billboard.
    #                 3. Perform OCR on the selected largest banner/signage/billboard.
    #                 4. Ignore all advertisements, promotional text, product brand in banner/signage/billboard.
    #                 5. Extract ONLY the store name from the banner/signage/billboard.
    #                 6. Return the result in exactly this format (do not modify the format):
    #                 {result: "STORE_NAME"}
    #
    #                 If no store name is found:
    #                 {result: "null"}
    #
    #                 Important:
    #                 - Return ONLY the exact store name as seen on the banner/signage
    #                 - Do not include any steps, analysis, or additional text
    #                 - The response should contain ONLY the {result: "STORE_NAME"} format
    #             """,
    #             images=[decoded]
    #         )
    #
    #         content = response.get("response", "")
    #         store_name = self.extract_store_name(content) if content else ""
    #
    #         return {
    #             "result": store_name,
    #             "raw_response": None if store_name else content
    #         }
    #
    #     except Exception as e:
    #         logger.error(f"Error in process_image_with_ollama: {e}")
    #         return {
    #             "error": "OCR processing failed",
    #             "details": str(e)
    #         }

    def extract_store_name(self, response_text: str) -> str:
        match = re.search(r"\{result:\s*\"([^\"]+)\"\}", response_text)
        return match.group(1).strip() if match else "null"

    async def update_ocr_store_names(self, pool, reffid, ocr_results):
        store_names = []
        for r in ocr_results[:3]:
            val = r.get("ocr_result")
            store_names.append(None if val in (None, "null", "") else val)
        while len(store_names) < 3:
            store_names.append(None)

        ocr_processed_date = datetime.datetime.now()

        query = """
            UPDATE public.mpos_company_profile
            SET ocr_store_name_1 = $1,
                ocr_store_name_2 = $2,
                ocr_store_name_3 = $3,
                ocr_processed_date = $4
            WHERE reffid = $5
        """
        async with pool.acquire() as conn:
            await conn.execute(query, store_names[0], store_names[1], store_names[2], ocr_processed_date, reffid)

    async def update_blur_scores(self, pool, reffid, blur_results):
        blur_scores = []
        for r in blur_results[:3]:
            val = r.get("blur_result", {}).get("quality_level")
            blur_scores.append(None if not val else val)
        while len(blur_scores) < 3:
            blur_scores.append(None)

        blur_processed_date = datetime.datetime.now()
        query = """
            UPDATE public.mpos_company_profile
            SET blur_score_1 = $1,
                blur_score_2 = $2,
                blur_score_3 = $3,
                blur_processed_date = $4
            WHERE reffid = $5
        """
        async with pool.acquire() as conn:
            await conn.execute(query, blur_scores[0], blur_scores[1], blur_scores[2], blur_processed_date, reffid)

    def detect_image_blur(self, image_path: str) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        h, w = gray.shape
        fft_shift[h // 2 - 30:h // 2 + 30, w // 2 - 30:w // 2 + 30] = 0
        recon = np.fft.ifft2(np.fft.ifftshift(fft_shift))
        fft_score = np.mean(20 * np.log(np.abs(recon) + 1e-8))
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_score = np.mean(np.sqrt(sobel_x ** 2 + sobel_y ** 2))
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2))

        thresholds = {
            'lap_low': 100, 'lap_high': 1000,
            'fft_low': 20, 'fft_high': 50,
            'sobel_low': 20, 'sobel_high': 50,
            'grad_low': 10, 'grad_high': 30
        }

        def normalize(value, low, high):
            if value >= high:
                return 100.0
            if value <= low:
                return 0.0
            return ((value - low) / (high - low)) * 100.0

        lap_n = normalize(laplacian_var, thresholds['lap_low'], thresholds['lap_high'])
        fft_n = normalize(fft_score, thresholds['fft_low'], thresholds['fft_high'])
        sobel_n = normalize(sobel_score, thresholds['sobel_low'], thresholds['sobel_high'])
        grad_n = normalize(gradient_magnitude, thresholds['grad_low'], thresholds['grad_high'])

        weights = {'laplacian': 0.4, 'fft': 0.3, 'sobel': 0.2, 'gradient': 0.1}
        sharpness_score = (
                lap_n * weights['laplacian'] +
                fft_n * weights['fft'] +
                sobel_n * weights['sobel'] +
                grad_n * weights['gradient']
        )

        if sharpness_score >= 70:
            classification = 'sharp'
            confidence = min(100.0, sharpness_score)
        elif sharpness_score <= 30:
            classification = 'blur'
            confidence = min(100.0, 100 - sharpness_score)
        else:
            classification = 'moderate_sharp' if sharpness_score > 50 else 'moderate_blur'
            confidence = abs(sharpness_score - 50) * 2

        blur_pct = 100 - confidence if classification in ('sharp', 'moderate_sharp') else confidence
        sharp_pct = confidence if classification in ('sharp', 'moderate_sharp') else 100 - confidence

        return {
            'raw_scores': {
                'laplacian': round(laplacian_var, 2),
                'fft': round(fft_score, 2),
                'sobel': round(sobel_score, 2),
                'gradient': round(gradient_magnitude, 2)
            },
            'normalized_scores': {
                'laplacian': round(lap_n, 2),
                'fft': round(fft_n, 2),
                'sobel': round(sobel_n, 2),
                'gradient': round(grad_n, 2)
            },
            'sharpness_score': round(sharpness_score, 2),
            'classification': classification,
            'confidence': round(confidence, 2),
            'percentages': {
                'sharp': round(sharp_pct, 2),
                'blur': round(blur_pct, 2)
            },
            'quality_level': self._get_quality_level(sharpness_score)
        }

    def _get_quality_level(self, score: float) -> str:
        if score >= 85:
            return 'excellent'
        if score >= 70:
            return 'good'
        if score >= 50:
            return 'fair'
        if score >= 30:
            return 'poor'
        return 'very_poor'

    def process_image_with_ollama(self, image_base64: str) -> dict:
        try:
            # Strip data URI prefix if present
            if image_base64.startswith('data:image'):
                base64_data = image_base64.split(',', 1)[1]
            else:
                base64_data = image_base64

            # Debug log: print raw Base64 length
            print("base64 length from mongo:", len(base64_data))

            # Call Ollama with Base64 inline (as required by llava)
            response = self.ollama_client.generate(
                model="llava",

                prompt="""
                You are an expert OCR agent specialized in store and restaurant signage. Given one image, follow these steps exactly:

                0. First, determine if the image contains a sign for a store, restaurant, food stall, street‐vendor cart, kiosk, or similar.  
                1. Detect all text regions in the image.  
                2. From those, choose the single region that is most prominent (largest text, highest contrast, top or center).  
                3. Perform OCR on that region—**always return its text**.  
                4. Within that text, look first for a store or restaurant name (shop sign, storefront banner, food stall name, cart name, kiosk name).  
                5. If you find a store or restaurant name, output it.  

                Return exactly one JSON object, no extra text or fields:

                - If you extracted any text (name or banner):  
                `{result: "THE_EXTRACTED_TEXT"}`  

                    **Examples:**  
                    - Image with “WARUNG SARI RASA” sign → `{result: "WARUNG SARI RASA"}`  
                    - Food cart header “Bakso Pak Joko” → `{result: "Bakso Pak Joko"}`  
                    - Restaurant awning “RUMAH MAKAN PADANG” → `{result: "RUMAH MAKAN PADANG"}`  
                    - Kiosk with banner “DRP” → `{result: "DRP"}`  
                """,
                # Send Base64 string, not raw bytes
                images=[base64_data]
        )

            # Extract response text
            content = response.get("response", "")
            store_name = self.extract_store_name(content) if content else ""

            return {
                "result": store_name
            }

        except Exception as e:
            logger.error(f"Error in process_image_with_ollama: {e}")
            return {
                "error": "OCR processing failed",
                "details": str(e)
            }

    async def process_image_with_ollama_from_file(self, image_file: UploadFile) -> dict:
        """
        Wrapper: baca UploadFile, ubah ke base64 string, lalu panggil
        process_image_with_ollama() yang sudah ada (tanpa ubah prompt).
        """
        # 1. Baca semua byte dari file
        img_bytes = await image_file.read()

        # 2. Encode ke Base64 dan tambahkan prefix data URI
        b64 = base64.b64encode(img_bytes).decode()
        image_base64 = f"data:image/jpeg;base64,{b64}"

        # 3. Panggil fungsi Anda yang lama—sama persis,
        #    hanya inputnya sekarang Base64 hasil pembacaan file
        return self.process_image_with_ollama(image_base64)
        # return self.process_image_with_ollama_chat_image(image_base64)



    async def process_image_llava(self, image_file: UploadFile) -> dict:
        """
        1. Baca UploadFile → bytes
        2. Encode ke Base64 (tanpa prefix)
        3. Panggil self.client.generate(...) dengan images=[base64]
        4. Parse response
        """
        try:
            # 1. Baca bytes dari UploadFile
            img_bytes = await image_file.read()

            # 2. Encode Base64
            b64 = base64.b64encode(img_bytes).decode()

            # 3. Panggil LLava via ollama.Client
            response = self.ollama_client.generate(
                model = "llava",
                prompt = """
Analyze the image and extract store information following these steps:  
1. Identify all banners/signage/billboards in the image.  
2. Detect and select the largest/most prominent banner or signage or billboard.  
3. Perform OCR on the selected largest banner/signage/billboard.
4. Ignore all advertisements, promotional text, product brand in banner/signage/billboard.
5. Extract ONLY the store name from the banner/signage/billboard.
6. Return the result in exactly this format (do not modify the format):
{result: "STORE_NAME"}

If no store name is found:
{result: "null"}

Important:
- Return ONLY the exact store name as seen on the banner/signage
- Do not include any steps, analysis, or additional text
- The response should contain ONLY the {result: "STORE_NAME"} format
""",
                images = [b64]
            )

            # 4. Ambil teks response
            # Ollama-Python .generate() biasanya kembalikan {"response": "..."}
            content = response.get("response", "")
            store_name = self.extract_store_name(content)

            return {
                "result": store_name,
                "raw_response": None if store_name != "null" else content
            }

        except Exception as e:
            logger.error(f"Error in LLavaProcessor.process_image: {e}", exc_info=True)
            return {
                "error": "OCR processing failed",
                "details": str(e)
            }
