import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to('cpu')
midas.eval()

# Load the transforms for MiDaS
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Read the input image
rgb_image_path = "/Users/garimasaroj/Downloads/openpose_example-main/distance_images/ima4.jpeg"
rgb_image = cv2.imread(rgb_image_path)

# Apply the transform to the image
input_batch = transform(rgb_image).to('cpu')

# Perform depth estimation
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=rgb_image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# Convert the prediction to a NumPy array
depth_map = prediction.cpu().numpy()

# Normalize the depth map for better visualization
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX)
depth_map_normalized = cv2.applyColorMap(depth_map_normalized.astype('uint8'), cv2.COLORMAP_JET)

# Overlay the depth map on the RGB image
overlay_image = cv2.addWeighted(rgb_image, 0.7, depth_map_normalized, 0.3, 0)

# Save the overlaid image
cv2.imwrite("overlay_image_with_depth.jpg", overlay_image)

# Display the overlay
plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

