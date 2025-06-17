import base64
import tempfile

def decode_image_base64_to_tempfile(image_base64: str) -> str:
    image_data = base64.b64decode(image_base64)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(image_data)
    temp_file.close()
    return temp_file.name
