from PIL import Image
import exifread

# Load the image
####################################### change this########################################
image_path = "dataset/xiaomi/xiaomi_xmouse60.jpg"  # Update with your image path
image = Image.open(image_path)

# Get image width
image_width = image.width
image_height = image.height 

# Open the image file for reading EXIF data
with open(image_path, 'rb') as img_file:
    tags = exifread.process_file(img_file)
    focal_length_raw = tags.get('EXIF FocalLength', 'Not available')
    focal_length_35mm_raw = tags.get('EXIF FocalLengthIn35mmFilm', 'Not available')
    
    # Convert focal length to float
    if focal_length_raw != 'Not available':
        focal_length = float(focal_length_raw.values[0])
    else:
        focal_length = 'Not available'
    
    # Convert 35mm focal length to int
    if focal_length_35mm_raw != 'Not available':
        focal_length_35mm = int(focal_length_35mm_raw.values[0])
    else:
        focal_length_35mm = 'Not available'

print(f"Image Width: {image_width}")
print(f"Image Height: {image_height}")
print(f"Focal Length: {focal_length} mm")
print(f"Focal Length (35mm equivalent): {focal_length_35mm} mm")

def calculate_object_size(pixel_length, image_width_pixels, sensor_width_mm, focal_length_mm, distance_mm):
    """
    Calculate real-world object size from pixel measurements.
    
    Formula:
    1. pixel_size_mm = sensor_width_mm / image_width_pixels
    2. object_size_on_sensor_mm = 
    th x pixel_size_mm
    3. real_object_size_mm = (object_size_on_sensor_mm x distance_mm) / focal_length_mm
    
    Args:
        pixel_length: Length of object in pixels
        image_width_pixels: Image width in pixels
        sensor_width_mm: Camera sensor width in mm
        focal_length_mm: Focal length in mm
        distance_mm: Distance from camera to object in mm
    
    Returns:
        Real object size in mm
    """
    # Calculate pixel size in mm
    pixel_size_mm = sensor_width_mm / image_width_pixels
    
    # Calculate object size on sensor
    object_size_on_sensor_mm = pixel_length * pixel_size_mm
    
    # Calculate real object size using similar triangles
    real_object_size_mm = (object_size_on_sensor_mm * distance_mm) / focal_length_mm
    
    return real_object_size_mm

# Your data
image_width_pixels = image_width # Image width in pixels or height
focal_length_mm = focal_length
####################################### change this########################################
sensor_width_mm = 5.76 # Example: Xiaomi Mi 10 sensor width or height in mm
pixel_length = 595 # Measured length in pixels from the image

# You need to know the distance to the object
distance_mm = 600  # Example: 1000mm = 1 meter (you need to measure this!)

object_size_mm = calculate_object_size(
    pixel_length=pixel_length,
    image_width_pixels=image_width_pixels,
    sensor_width_mm=sensor_width_mm,
    focal_length_mm=focal_length_mm,
    distance_mm=distance_mm
)

print(f"Real object size: {object_size_mm:.2f} mm ({object_size_mm/10:.2f} cm)")

