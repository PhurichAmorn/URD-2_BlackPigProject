from PIL import Image, ExifTags
import math

def get_exif_data(image_path):
    """Extract EXIF metadata from image."""
    img = Image.open(image_path)
    exif_data = {}
    if hasattr(img, "_getexif"):
        exif = img._getexif()
        if exif:
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                exif_data[decoded] = value
    return exif_data, img.size  # (metadata, (width_px, height_px))

def get_pixel_size(exif_data, img_size, sensor_width_mm=6.4):
    """
    Compute size of one pixel in mm using EXIF + sensor info.
    Default sensor width ~6.4mm (common for smartphone cameras).
    """
    width_px, _ = img_size

    # Extract focal length (in mm)
    focal_length = None
    if "FocalLength" in exif_data:
        fl = exif_data["FocalLength"]
        focal_length = fl[0] / fl[1] if isinstance(fl, tuple) else float(fl)

    # User-specified subject distance (in mm)
    subject_distance = None
    # Try EXIF first
    if "SubjectDistance" in exif_data:
        sd = exif_data["SubjectDistance"]
        subject_distance = sd[0] / sd[1] if isinstance(sd, tuple) else float(sd)

    # If not in EXIF, ask user to specify (example: 1500mm = 1.5m)
    if not subject_distance:
        subject_distance = 1500  # <-- Set your actual distance here in mm

    if not focal_length:
        raise ValueError("No focal length found in EXIF data.")

    # Pixel pitch (mm per pixel on sensor)
    pixel_pitch = sensor_width_mm / width_px

    # mm per pixel at subject plane (accurate)
    mm_per_pixel_subject = (subject_distance * pixel_pitch) / focal_length
    # mm per pixel at sensor (approximate)
    mm_per_pixel_sensor = pixel_pitch

    return mm_per_pixel_subject, mm_per_pixel_sensor

def measure_object_length(pixel_length, mm_per_pixel):
    """Convert pixel measurement into real-world length (cm)."""
    return (pixel_length * mm_per_pixel) / 10.0  # convert mm → cm


# ================================

# Example usage
# ================================

image_path = "blackpig.jpg"  # your image file
exif_data, img_size = get_exif_data(image_path)
mm_per_pixel_subject, mm_per_pixel_sensor = get_pixel_size(exif_data, img_size)

# Example: Suppose segmentation/bounding box says object is 1200 px long
pixel_length = 1200
real_length_cm_subject = measure_object_length(pixel_length, mm_per_pixel_subject)
real_length_cm_sensor = measure_object_length(pixel_length, mm_per_pixel_sensor)

print(f"Object length (using subject distance): {real_length_cm_subject:.2f} cm")
print(f"Object length (using sensor only): {real_length_cm_sensor:.2f} cm")
