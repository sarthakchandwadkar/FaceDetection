from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys
import matplotlib.pyplot as plt
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

def main():
    # Load environment variables
    load_dotenv()
    ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
    ai_key = os.getenv('AI_SERVICE_KEY')

    # Initialize the Computer Vision client
    credentials = CognitiveServicesCredentials(ai_key)
    cv_client = ComputerVisionClient(ai_endpoint, credentials)

    # Get image file
    image_file = 'images/people.jpg'
    if len(sys.argv) > 1:
        image_file = sys.argv[1]

    # Analyze image
    analyze_image(image_file, cv_client)

def analyze_image(image_file, cv_client):
    print('\nAnalyzing', image_file)

    # Open the image file
    with open(image_file, "rb") as image:
        # Analyze image for objects
        analysis = cv_client.analyze_image_in_stream(image, visual_features=[VisualFeatureTypes.objects])

    # Check for detected objects
    if analysis.objects:
        print("\nObjects in image:")
        for detected_object in analysis.objects:
            if detected_object.object_property == "person":
                # Use the 'rectangle' attribute to get the bounding box
                bounding_box = detected_object.rectangle
                print(f"Detected person with bounding box: {bounding_box}, Confidence: {detected_object.confidence:.2f}")
    else:
        print("No objects detected.")


if __name__ == "__main__":
    main()
