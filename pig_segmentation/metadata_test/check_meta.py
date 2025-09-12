from PIL import Image

img = Image.open('blackpig1.jpg')
exif_data = img._getexif()
if exif_data:
    print("EXIF data found.")
else:
    print("No EXIF data found.")