from PIL import Image
from PIL.ExifTags import TAGS

def print_exif(image_path):
    with Image.open(image_path) as img:
        exif_data = img._getexif()
        if exif_data is None:
            print("No EXIF data found.")
            return
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            print(f"{tag}: {value}")

if __name__ == "__main__":
    image_path = "blackpig.jpg"  # Change to your image file path
    print_exif(image_path)