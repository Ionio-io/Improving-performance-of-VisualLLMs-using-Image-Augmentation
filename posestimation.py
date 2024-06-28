import cv2 as cv
import numpy as np
from openai import OpenAI
import os
import base64
import requests
import matplotlib.pyplot as plt
import argparse
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

image_width, image_height = 600, 600
threshold = 0.2

def run_pose_estimation(image_path, model_path="graph_opt.pb"):
    net = cv.dnn.readNetFromTensorflow(model_path)
    img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    if img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    
    photo_height, photo_width = img.shape[:2]
    net.setInput(cv.dnn.blobFromImage(img, 1.0, (image_width, image_height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (photo_width * point[0]) / out.shape[3]
        y = (photo_height * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(img, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(img, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(img, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    
    return img

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_gpt4v_critique(image_path, client_instance, with_pose_lines=True, retries=3):
    base64_image = encode_image(image_path)
    critique_prompt = "Act as my gym trainer and examine the image of me performing this exercise. Please provide detailed feedback on how I can improve my posture and technique. If there are any specific issues with my alignment or positioning, highlight them and suggest corrections."
    if with_pose_lines:
        critique_prompt += " Use the pose estimation lines overlayed on the image to guide your critique."

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful and knowledgeable gym trainer."
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

    for attempt in range(retries):
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()
        if response.status_code == 200:
            return response_data["choices"][0]["message"]["content"]
        else:
            print(f"Attempt {attempt + 1} failed: {response.status_code} - {response_data}")

    raise Exception(f"Failed to get critique after {retries} attempts: {response.status_code} - {response_data}")

def compare_critiques(image_path, initial_critique, enhanced_critique, client_instance):
    base64_image = encode_image(image_path)
    comparison_prompt = f"Which critique is more accurate or helpful for scenic description?\n\nInitial Critique: {initial_critique}\n\nEnhanced Critique: {enhanced_critique}"
    response = client_instance.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": comparison_prompt},
            {"role": "user", "content": f"data:image/jpeg;base64,{base64_image}"}
        ],
        max_tokens=150
    )
    return response.choices[0].message["content"].strip()

def benchmark_critique(image_paths, client_instance):
    results = []
    for image_path in image_paths:
        initial_critique = get_gpt4v_critique(image_path, client_instance, with_pose_lines=False)
        overlay_image = run_pose_estimation(image_path)
        overlay_image_path = image_path.replace(".jpg", "_overlay.jpg")
        cv.imwrite(overlay_image_path, overlay_image)
        
        enhanced_critique = get_gpt4v_critique(overlay_image_path, client_instance, with_pose_lines=True)
        
        comparison_result = compare_critiques(initial_critique, enhanced_critique, client_instance)
        
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

def plot_comparison_results(benchmark_results):
    num_images = len(benchmark_results)
    initial_better_count = sum(1 for result in benchmark_results if "initial" in result["comparison_result"].lower())
    enhanced_better_count = num_images - initial_better_count
    
    labels = ["Initial Critique", "Enhanced Critique"]
    counts = [initial_better_count, enhanced_better_count]
    
    plt.bar(labels, counts, color=['blue', 'green'])
    plt.xlabel('Critique Type')
    plt.ylabel('Number of Images')
    plt.title('Comparison of Critiques')
    plt.show()

# Main function to orchestrate the entire process
def main(args):
    image_paths = get_image_paths_from_folder(args.image_folder)
    benchmark_results = benchmark_critique(image_paths, client)
    
    for result in benchmark_results:
        print(f"Image: {result['image_path']}")
        print(f"Initial Critique: {result['initial_critique']}")
        print(f"Enhanced Critique: {result['enhanced_critique']}")
        print(f"Comparison Result: {result['comparison_result']}")
        print("\n")
    
    plot_comparison_results(benchmark_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarking of image critiques with GPT-4 and visualize results.")
    parser.add_argument("image_folder", type=str, help="Path to folder containing images for critique.")

    args = parser.parse_args()
    main(args)