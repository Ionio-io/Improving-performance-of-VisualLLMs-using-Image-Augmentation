import os
import cv2 as cv
import numpy as np
import base64
import requests
import matplotlib.pyplot as plt
import argparse
from ultralytics import YOLO
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_yolov8(image_path, model_path="yolov8n.pt", target_size=(640, 480)):
    model = YOLO(model_path)
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Unable to read image at {image_path}")

    # Resize image
    img_resized = cv.resize(img, target_size)
    results = model(img_resized)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {confidence:.2f}"
            cv.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(img_resized, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    enhanced_image_path = get_overlay_image_path(image_path)
    cv.imwrite(enhanced_image_path, img_resized)
    
    return img_resized

def encode_image(image_path, target_size=(320, 240)):  # Further reduce the size
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Unable to read image at {image_path}")
    
    img_resized = cv.resize(img, target_size)
    _, buffer = cv.imencode('.jpeg', img_resized)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    return base64_image

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_gpt4_critique(image_path, client_instance, with_bounding_boxes=True):
    base64_image = encode_image(image_path)
    critique_prompt = "Describe the whole scene with all objects and elements present in the image."
    if with_bounding_boxes:
        critique_prompt += " Use the bounding boxes overlayed on the image to guide your description."

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful and knowledgeable assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": critique_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client_instance.api_key}"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Failed to get critique: {response.status_code} - {response.text}")

def compare_critiques(image_path, initial_critique, enhanced_critique, client_instance):
    base64_image = encode_image(image_path)
    comparison_prompt = f"Which critique is more accurate or helpful for scenic description?\n\nInitial Critique: {initial_critique}\n\nEnhanced Critique: {enhanced_critique}"
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful and knowledgeable assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": comparison_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 200
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client_instance.api_key}"
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"Failed to compare critiques: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Failed to compare critiques: {e}")
        return None

def benchmark_critique(image_paths, client_instance):
    results = []
    for image_path in image_paths:
        initial_image_path = image_path
        enhanced_image = run_yolov8(image_path)
        enhanced_image_path = get_overlay_image_path(image_path)
        
        initial_critique = get_gpt4_critique(initial_image_path, client_instance, with_bounding_boxes=False)
        enhanced_critique = get_gpt4_critique(enhanced_image_path, client_instance, with_bounding_boxes=True)
        
        comparison_result = compare_critiques(initial_image_path, initial_critique, enhanced_critique, client_instance)
        
        results.append({
            "image_path": image_path,
            "initial_critique": initial_critique,
            "enhanced_critique": enhanced_critique,
            "comparison_result": comparison_result
        })
        
    return results

def get_image_paths_from_folder(folder_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(image_extensions)]

def get_overlay_image_path(image_path):
    base, _ = os.path.splitext(image_path)
    return f"{base}_overlay.jpeg"

def plot_comparison_results(benchmark_results):
    num_images = len(benchmark_results)
    initial_better_count = sum(1 for result in benchmark_results if result["comparison_result"] and "initial" in result["comparison_result"].lower())
    enhanced_better_count = num_images - initial_better_count
    
    labels = ["Initial Critique", "Enhanced Critique"]
    counts = [initial_better_count, enhanced_better_count]
    
    plt.bar(labels, counts, color=['blue', 'green'])
    plt.xlabel('Critique Type')
    plt.ylabel('Number of Images')
    plt.title('Comparison of Critiques')
    plt.show()

def main(args):
    image_paths = get_image_paths_from_folder(args.image_folder)
    benchmark_results = benchmark_critique(image_paths, client)
    
    for result in benchmark_results:
        print(f"Image: {result['image_path']}")
        print("\n")
        print(f"Initial Critique: {result['initial_critique']}")
        print("\n")
        print(f"Enhanced Critique: {result['enhanced_critique']}")
        print("\n")
        print(f"Comparison Result: {result['comparison_result']}")
        print("\n")
    
    plot_comparison_results(benchmark_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarking of image critiques with YOLOv8 and GPT-4 and visualize results.")
    parser.add_argument("image_folder", type=str, help="Path to folder containing images for critique.")

    args = parser.parse_args()
    main(args)
