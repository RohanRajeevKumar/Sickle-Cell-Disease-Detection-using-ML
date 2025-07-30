from PIL import Image, ImageEnhance

# Load the source image
source_path = "enhanced.jpg"
output_path = "zoomed.jpg"
zoom_factor = 1.5  # Adjust this for zoom level (e.g., 1.2 means 20% zoom)

# Open the image
source_img = Image.open(source_path)
width, height = source_img.size

# Calculate cropping box for zooming
crop_width = int(width / zoom_factor)
crop_height = int(height / zoom_factor)
left = (width - crop_width) // 2
top = (height - crop_height) // 2
right = left + crop_width
bottom = top + crop_height

# Crop the image
cropped_img = source_img.crop((left, top, right, bottom))

# Resize the cropped image back to the original dimensions
zoomed_img = cropped_img.resize((width, height), Image.Resampling.LANCZOS)

# Save the zoomed image
zoomed_img.save(output_path, "JPEG", quality=90)

print(f"Zoomed image saved at: {output_path}")
