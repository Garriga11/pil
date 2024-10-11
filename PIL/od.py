from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2








# Step 1: Load 5 images from file paths
image_paths = [
    "images/bentRim.jpg",
    "images/radialCracking.jpg",
    
    
]

# Load the images and convert them to NumPy arrays (grayscale for simplicity)
images = [np.array(Image.open(path).convert('L')) for path in image_paths]

# Step 2: Define the ground truth bounding boxes (xmin, ymin, xmax, ymax)
bounding_boxes = [
    np.array([50, 50, 400, 300]),  # Image 1
    np.array([200, 200, 700, 600]),  # Image 2

]

# Step 3: Visualize and save the images with bounding boxes
for i, (image, box) in enumerate(zip(images, bounding_boxes)):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    
    # Create a rectangle patch for the bounding box
    rect = patches.Rectangle(
        (box[0], box[1]),  # (xmin, ymin)
        box[2] - box[0],   # width
        box[3] - box[1],   # height
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.set_title(f"Image {i + 1}")
    ax.axis('off')

    # Save the image with bounding box
    plt.savefig(f"image_with_box_{i + 1}.png", bbox_inches='tight')
    plt.close()  # Close the figure to avoid displaying it
# Example predicted bounding box from your model 
predicted_box = [60, 60, 90, 90]  # (x, y, width, height)

# Manually defined ground truth bounding box
ground_truth_box = [50, 50, 100, 100]  # (x, y, width, height)

# Function to calculate IoU
def calculate_iou(box1, box2):
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2

    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2

    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# Calculate IoU between predicted and ground truth boxes
iou_value = calculate_iou(predicted_box, ground_truth_box)
print(f"IoU: {iou_value}") 





def iou(box1, box2): 
    # Calculate the (x, y) coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # Calculate the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate the area of both bounding boxes
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate IoU
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

# Example usage with your bounding box and a ground truth box
iou_score = iou(bounding_boxes[0], ground_truth_box) # type: ignore
print(f"IoU Score for Image 1: {iou_score}") 



# Load and preprocess your images
def load_images(image_paths, target_size=(224, 224)):
    images = []
    for img in image_paths:
        img = cv2.imread("images")
        img = cv2.resize(image, target_size)  # Resize to match the input shape of the model
        img = img / 255.0  # Normalize to 0-1 range
        images.append(img)
    return np.array(images)