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

# Initialize MongoDB client (tidak berubah)
mongo_client = MongoClient(settings.MONGO_URI)
mongo_db = mongo_client[settings.MONGO_DB]
mongo_collection = mongo_db[settings.MONGO_COLLECTION]


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

            prompt = ("""
Analyze the image and extract store information, including small street vendors and roadside stalls (toko kaki lima):

1. Search for ANY visible text that could indicate a business name, including:
   - Large storefront signs and banners
   - Small handwritten signs on stalls or carts
   - Text on umbrellas, tents, or temporary structures
   - Names written on wooden boards, cardboard, or plastic signs
   - Text on food carts, mobile vendors, or street stalls
   - Names painted or written on walls, doors, or windows
   - Text on fabric banners or cloth signs
   - Even partial or faded text that might be readable

2. Look specifically for street vendor indicators:
   - Food cart names (e.g., "BAKSO PAK JOKO", "ES TEH MANIS IBU SARI")
   - Stall names (e.g., "WARUNG NASI GUDEG", "KOPI ANGKRINGAN")
   - Vendor names (e.g., "GORENGAN BU TINI", "RUJAK CINGUR PAK WAHYU")
   - Product + owner names (e.g., "MIE AYAM BUDI", "SATE KAMBING HAJI ALI")

3. Consider these text locations for kaki lima:
   - On pushcarts or mobile stalls
   - Handwritten signs on small boards
   - Text on plastic tarps or cloth coverings
   - Names on cooking equipment or containers
   - Signs attached to motorcycles or bicycles
   - Text on small umbrellas or parasols

4. Text recognition guidelines:
   - Accept handwritten text, even if not perfectly clear
   - Include names with titles (Pak, Bu, Ibu, Bapak, Haji, etc.)
   - Accept mixed language (Indonesian + local dialect)
   - Consider abbreviated names (e.g., "WARTEG" for "Warung Tegal")

5. Return null ONLY if:
   - Image is completely unreadable (too blurry, dark, or corrupted)
   - Absolutely no text or signage exists anywhere
   - Image shows only products without any identifying text

Return the result in exactly this format:
{result: "STORE_NAME"}

Examples for kaki lima:
- Handwritten "Bakso Pak Joko": {result: "BAKSO PAK JOKO"}
- Small sign "Warteg Bahari": {result: "WARTEG BAHARI"}
- Text on cart "Es Teh Manis": {result: "ES TEH MANIS"}
- Faded text "Nasi Gud_g Bu S_ri" (readable as Gudeg): {result: "NASI GUDEG BU SARI"}
- Sign "Kopi Angkringan": {result: "KOPI ANGKRINGAN"}

Important:
- Be very inclusive - any readable business-related text counts
- Don't ignore small, handwritten, or informal signage
- Street vendors often have simple, direct naming
- Include food type + owner name combinations
- Return ONLY the {result: "STORE_NAME"} format
""")

            # 4. Call module-level API (supports images=[])
            response = self.ollama_client.chat(
                model=settings.OLLAMA_MODEL,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [tmp_path]
                }],
            )
            os.unlink(tmp_path)
            # 6. Parse the first choice’s content
            choices = response.get('choices', [])
            content = ""
            if choices:
                content = choices[0].get('message', {}).get('content', "")
            else:
                # fallback if the API returns a top-level 'response'
                content = response.get('response', "")
            store_name = self.extract_store_name(content)

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

    # async def process_image_with_ollama(self, image_base64: str) -> dict:
    #     try:
    #         if image_base64.startswith('data:image'):
    #             base64_data = image_base64.split(',', 1)[1]
    #         else:
    #             base64_data = image_base64

    #         with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    #             tmp.write(base64.b64decode(base64_data))
    #             tmp_path = tmp.name

    #         response = self.ollama_client.generate(
    #             model="llava",
    #             prompt = ("""
    #                 Analyze the image and extract store information following these steps:
    #                 1. Identify all banners/signage/billboards in the image.
    #                 2. Detect and select the largest/most prominent banner or signage or billboard.
    #                 3. Perform OCR on the selected largest banner/signage/billboard.
    #                 4. Ignore all advertisements, promotional text, product brand in banner/signage/billboard.
    #                 5. Extract ONLY the store name from the banner/signage/billboard.
    #                 6. Return the result in exactly this format (do not modify the format):
    #                 {result: "STORE_NAME"}

    #                 Examples:
    #                 For a store "BUDI JAYA":
    #                 {result: "BUDI JAYA"}

    #                 For a store "TOKO BERKAH":
    #                 {result: "TOKO BERKAH"}

    #                 If no store name is found:
    #                 {result: "null"}

    #                 Important:
    #                 - Return ONLY the exact store name as seen on the banner/signage
    #                 - Do not include any steps, analysis, or additional text
    #                 - The response should contain ONLY the {result: "STORE_NAME"} format
    #             """
    #         ),

    #             images=[tmp_path]
    #         )

    #         os.unlink(tmp_path)

    #         content = response.get("response", "")
    #         store_name = self.extract_store_name(content) if content else ""

    #         return {
    #             "result": store_name,
    #             "raw_response": content if not store_name else None
    #         }

    #     except Exception as e:
    #         logger.error(f"Error in process_image_with_ollama: {str(e)}")
    #         return {
    #             "error": "OCR processing failed",
    #             "details": str(e)
    #         }

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

            # 3. Prompt tetap sama (tidak diubah)
            prompt = """
    Analyze the image and extract store information, including small street vendors and roadside stalls (toko kaki lima):

    1. Search for ANY visible text that could indicate a business name, including:
       - Large storefront signs and banners
       - Small handwritten signs on stalls or carts
       - Text on umbrellas, tents, or temporary structures
       - Names written on wooden boards, cardboard, or plastic signs
       - Text on food carts, mobile vendors, or street stalls
       - Names painted or written on walls, doors, or windows
       - Text on fabric banners or cloth signs
       - Even partial or faded text that might be readable

    2. Look specifically for street vendor indicators:
       - Food cart names (e.g., "BAKSO PAK JOKO", "ES TEH MANIS IBU SARI")
       - Stall names (e.g., "WARUNG NASI GUDEG", "KOPI ANGKRINGAN")
       - Vendor names (e.g., "GORENGAN BU TINI", "RUJAK CINGUR PAK WAHYU")
       - Product + owner names (e.g., "MIE AYAM BUDI", "SATE KAMBING HAJI ALI")

    3. Consider these text locations for kaki lima:
       - On pushcarts or mobile stalls
       - Handwritten signs on small boards
       - Text on plastic tarps or cloth coverings
       - Names on cooking equipment or containers
       - Signs attached to motorcycles or bicycles
       - Text on small umbrellas or parasols

    4. Text recognition guidelines:
       - Accept handwritten text, even if not perfectly clear
       - Include names with titles (Pak, Bu, Ibu, Bapak, Haji, etc.)
       - Accept mixed language (Indonesian + local dialect)
       - Consider abbreviated names (e.g., "WARTEG" for "Warung Tegal")

    5. Return null ONLY if:
       - Image is completely unreadable (too blurry, dark, or corrupted)
       - Absolutely no text or signage exists anywhere
       - Image shows only products without any identifying text

    Return the result in exactly this format:
    {result: "STORE_NAME"}

    Examples for kaki lima:
    - Handwritten "Bakso Pak Joko": {result: "BAKSO PAK JOKO"}
    - Small sign "Warteg Bahari": {result: "WARTEG BAHARI"}
    - Text on cart "Es Teh Manis": {result: "ES TEH MANIS"}
    - Faded text "Nasi Gud_g Bu S_ri" (readable as Gudeg): {result: "NASI GUDEG BU SARI"}
    - Sign "Kopi Angkringan": {result: "KOPI ANGKRINGAN"}

    Important:
    - Be very inclusive - any readable business-related text counts
    - Don't ignore small, handwritten, or informal signage
    - Street vendors often have simple, direct naming
    - Include food type + owner name combinations
    - Return ONLY the {result: "STORE_NAME"} format
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
                "result": store_name,
                "raw_response": None if store_name != "null" else content
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

    async def process_image_with_ollama(self, image_base64: str) -> dict:
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
                # Send Base64 string, not raw bytes
                images=[base64_data]
            )

            # Extract response text
            content = response.get("response", "")
            store_name = self.extract_store_name(content) if content else ""

            return {
                "result": store_name,
                "raw_response": None if store_name else content
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
        return await self.process_image_with_ollama(image_base64)



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
            response = self.client.generate(
                model = self.model,
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
