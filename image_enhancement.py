from PIL import Image, ImageEnhance
import cv2
import numpy as np

# Load the source image
source_path = "source.jpg"
output_path = "enhanced.jpg"

# Open the source image
source_img = Image.open(source_path)

# Resize the image to 380×285
source_img = source_img.resize((380, 285), Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality resizing


# Enhance the color
color_enhancer = ImageEnhance.Color(source_img)
source_img = color_enhancer.enhance(3.0)  # Increase color vibrancy significantly

# Enhance the contrast
contrast_enhancer = ImageEnhance.Contrast(source_img)
source_img = contrast_enhancer.enhance(2.0)  # Boost contrast further

# Sharpen the image
sharpness_enhancer = ImageEnhance.Sharpness(source_img)
source_img = sharpness_enhancer.enhance(3.0)  # Stronger sharpening

# Convert the image to numpy array for additional processing (e.g., brightness correction)
source_array = np.array(source_img)

# Adjust brightness and make background white using OpenCV
hsv = cv2.cvtColor(source_array, cv2.COLOR_RGB2HSV)
hsv[..., 1] = cv2.add(hsv[..., 1], -150)  # Reduce saturation slightly to emphasize red
hsv[..., 2] = cv2.add(hsv[..., 2], 80)  # Increase brightness significantly
source_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# Save the enhanced image with compression
enhanced_img = Image.fromarray(source_array)

# Save with high compression (quality parameter) to reduce file size
enhanced_img.save(output_path, "JPEG", quality=15, optimize=True)  # Adjust quality to control file size

print(f"Enhanced image saved at: {output_path}")
