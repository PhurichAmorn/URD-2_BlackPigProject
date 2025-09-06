import pillow_heif
import piexif

# Step 1: Open HEIC and extract metadata
heif_file = pillow_heif.open_heif("blackpig.HEIC", convert_hdr_to_8bit=False, bgr_mode=False)
image = heif_file[0].to_pillow()
exif_data = heif_file[0].info.get("exif", None)

# Step 2: Save image as JPEG
image.save("blackpig.jpg", format="JPEG")

# Step 3: Write metadata to JPEG (if available)
if exif_data:
    piexif.insert(exif_data, "blackpig.jpg")