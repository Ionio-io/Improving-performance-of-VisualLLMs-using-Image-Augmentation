import os
import cv2 as cv
import numpy as np
import base64
import requests
import matplotlib.pyplot as plt
import argparse
from dotenv import load_dotenv
import torch
from torchvision.transforms import functional as F
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to('cpu')
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

def run_depth_estimation(image_path):
    rgb_image = cv.imread(image_path)
    if rgb_image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    input_batch = transform(rgb_image).to('cpu')

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_map_normalized = cv.normalize(depth_map, None, 0, 255, norm_type=cv.NORM_MINMAX)
    depth_map_normalized = cv.applyColorMap(depth_map_normalized.astype('uint8'), cv.COLORMAP_JET)

    overlay_image = cv.addWeighted(rgb_image, 0.7, depth_map_normalized, 0.3, 0)

    return overlay_image, depth_map

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_gpt4v_critique(image_path, client_instance, with_depth_map=True, retries=3):
    base64_image = encode_image(image_path)
    critique_prompt = "Examine the image carefully and tell how distant are objects situated in there, considering if we're driving or navigating in that particular scenario."
    if with_depth_map:
        critique_prompt += " Consider the depth map overlaid in the image to judge the distance estimation in the image, note that colors represent distances: warmer colors (e.g., red) indicate closer objects, and cooler colors (e.g., blue) indicate farther objects."

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

    for attempt in range(retries):
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()
        if response.status_code == 200:
            return response_data["choices"][0]["message"]["content"]
        else:
            print(f"Attempt {attempt + 1} failed: {response.status_code} - {response_data}")

    raise Exception(f"Failed to get critique after {retries} attempts: {response.status_code} - {response_data}")

def compare_critiques(initial_critique, enhanced_critique, client_instance):
    comparison_prompt = f"Which critique is more accurate or helpful for estimating distances in the scene?\n\nInitial Critique: {initial_critique}\n\nEnhanced Critique: {enhanced_critique}"
    response = client_instance.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": comparison_prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def benchmark_critique(image_paths, client_instance):
    results = []
    for image_path in image_paths:
        initial_critique = get_gpt4v_critique(image_path, client_instance, with_depth_map=False)
        overlay_image, _ = run_depth_estimation(image_path)
        overlay_image_path = get_overlay_image_path(image_path)
        cv.imwrite(overlay_image_path, overlay_image)
        
        enhanced_critique = get_gpt4v_critique(overlay_image_path, client_instance, with_depth_map=True)
        
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

def get_overlay_image_path(image_path):
    base, ext = os.path.splitext(image_path)
    if ext.lower() in ['.jpg', '.jpeg']:
        return base + "_overlay" + ext
    else:
        return image_path.replace(ext, "_overlay" + ext)

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
