import os
import cv2
import numpy as np
import json
from PIL import Image

# Function to calculate the aspect ratio of an image
def get_aspect_ratio(image):
    width, height = image.size
    return width / height

# Function to find the best matching mockup template based on aspect ratio
def find_mockup_template(aspect_ratio, templates):
    closest_template = min(templates, key=lambda x: abs(aspect_ratio - x['aspect_ratio']))
    return closest_template

# Function to detect the green rectangle in the template and return its inner position
def detect_rectangle_position(mockup_path):
    # Load the mockup image using OpenCV
    mockup = cv2.imread(mockup_path)

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(mockup, cv2.COLOR_BGR2HSV)

    # Define the color range for detecting the green rectangle (adjust if necessary)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create a mask for the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the approximated contour has 4 points, we assume it's a rectangle
        if len(approx) == 4:
            # Get the bounding box coordinates
            x, y, w, h = cv2.boundingRect(approx)

            return (x, y, w, h)

    return None

# Function to create mockup
def create_mockup(user_image_path, mockup_template):
    user_image = Image.open(user_image_path)

    # Detect rectangle position in the mockup template
    rectangle_position = detect_rectangle_position(mockup_template['template_path'])

    if not rectangle_position:
        raise ValueError("No green rectangle found in the mockup template.")

    x, y, w, h = rectangle_position

    # Open mockup template using PIL
    mockup = Image.open(mockup_template['template_path'])

    # Resize user_image to fit within the detected rectangle
    user_image = user_image.resize((w, h), Image.LANCZOS)

    # Paste user_image onto mockup at the detected position
    mockup.paste(user_image, (x, y))

    # Save the result
    result_path = f"output_images/mockup_{os.path.basename(user_image_path)}"
    mockup.save(result_path)
    
    return result_path

# Main function to process the user's image and generate mockups
def generate_mockups(user_image_path, templates):
    user_image = Image.open(user_image_path)
    aspect_ratio = get_aspect_ratio(user_image)

    print(f"Aspect ratio of user image: {aspect_ratio}")
    
    # Find the appropriate mockup template
    mockup_template = find_mockup_template(aspect_ratio, templates)

    print(f"Selected mockup template: {mockup_template['template_path']}")
    
    # Create the mockup
    result_path = create_mockup(user_image_path, mockup_template)

    print(f"Mockup generated: {result_path}")
    
    # Return the result as a JSON object
    result = {
        "mockup_file": result_path,
    }
    
    return json.dumps(result)

if __name__ == "__main__":
    # Available mockup templates with their aspect ratios
    mockup_templates = [
        {"aspect_ratio": 1.33, "template_path": "mockup_images/green_1x1.png"}
    ]
    
    # User Images from Input
    user_image_path = "input_images/Ethiopian_woman_art.png"

    print(generate_mockups(user_image_path, mockup_templates))
