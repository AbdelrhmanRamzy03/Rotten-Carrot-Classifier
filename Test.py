import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    print("Image loaded successfully with shape:", img.shape)
    return img

def create_carrot_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_carrot = np.array([12, 180, 180])
    upper_carrot = np.array([18, 255, 255])
    mask_carrot = cv2.inRange(hsv, lower_carrot, upper_carrot)
    return mask_carrot

def create_rotten_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([40, 255, 130])
    hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    _, gray_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    combined = cv2.bitwise_or(hsv_mask, gray_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    return cleaned

def is_carrot_present(mask_carrot, img_shape):
    total_carrot_pixels = np.sum(mask_carrot > 0)
    carrot_ratio = total_carrot_pixels / (img_shape[0] * img_shape[1])
    return total_carrot_pixels > 1000 and 0.01 < carrot_ratio < 0.5

def classify_vegetable_freshness(img):
    carrot_mask = create_carrot_mask(img)
    rotten_mask = create_rotten_mask(img)

    rotten_inside_carrot = cv2.bitwise_and(carrot_mask, rotten_mask)

    total_carrot_pixels = np.sum(carrot_mask > 0)
    total_rotten_pixels = np.sum(rotten_inside_carrot > 0)

    rotten_ratio = total_rotten_pixels / total_carrot_pixels if total_carrot_pixels else 0

    if not is_carrot_present(carrot_mask, img.shape):
        return "Not a carrot", carrot_mask, rotten_inside_carrot

    if rotten_ratio > 0.25 and total_rotten_pixels > 500:
        return "Not Fresh", carrot_mask, rotten_inside_carrot
    else:
        return "Fresh", carrot_mask, rotten_inside_carrot

def display_result(img, rotten_mask, classification, use_matplotlib=False):
    output = img.copy()
    output[rotten_mask > 0] = [0, 0, 255]

    print(f"Classification Result: {classification}")

    if use_matplotlib:
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.imshow(output_rgb)
        plt.title(f"Rotten Carrots Detection - {classification}")
        plt.axis('off')
        plt.show()
    else:
        cv2.imshow("Original", img)
        cv2.imshow("Rotten Inside Carrot", rotten_mask)
        cv2.imshow(f"Rotten Carrots Detection - {classification}", output)
        print("Press any key on the image window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Open file dialog to select image
    Tk().withdraw()  # Hide the main tkinter window
    path = askopenfilename(title="Select a carrot image", filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
    
    if not path:
        print("No file selected.")
    else:
        img = load_image(path)
        classification, carrot_mask, rotten_mask = classify_vegetable_freshness(img)
        display_result(img, rotten_mask, classification)
