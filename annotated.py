import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color
import os

def extract_annotation_mask(annotated_image, annotation_color=[0, 0, 255]):
    """
    Extract a mask of pixels that are annotated with the specified color (default is red).
    
    Args:
        annotated_image: The input BGR image with annotations
        annotation_color: The BGR color used for annotation (default is red in BGR: [0, 0, 255])
        
    Returns:
        Mask of annotated pixels
    """
    # Convert annotation color to BGR (OpenCV format)
    annotation_color_bgr = np.array(annotation_color)
    # for maciecks picture
    annotation_color_bgr = np.array([43, 21, 229])
    
    # Create a mask for annotated pixels
    lower_bound = annotation_color_bgr - np.array([15, 15, 15])
    upper_bound = annotation_color_bgr + np.array([15, 15, 15])
    
    mask = cv2.inRange(annotated_image, lower_bound, upper_bound)
    return mask

def extract_pumpkin_pixels_using_mask(original_image, mask):
    """
    Extract the actual pumpkin pixels from the original (non-annotated) image using the annotation mask.
    
    Args:
        original_image: The original image without annotations in BGR format
        mask: Binary mask of annotated pixels
        
    Returns:
        RGB values of the pumpkin pixels from the original image
    """
    # Convert original image to RGB
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Extract RGB values of pixels corresponding to the annotation mask
    pumpkin_pixels = rgb_image[mask > 0]
    
    return pumpkin_pixels

def compute_color_statistics(pixels, color_space="RGB"):
    """
    Compute mean and standard deviation of pixel values in the specified color space.
    
    Args:
        pixels: Array of pixel values in RGB format
        color_space: Target color space ("RGB" or "CieLAB")
        
    Returns:
        pixel values in the specified color space, mean, and std
    """
    if color_space == "CieLAB":
        # Convert RGB to CieLAB
        lab_pixels = color.rgb2lab(pixels.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
        mean = np.mean(lab_pixels, axis=0)
        std = np.std(lab_pixels, axis=0)
        return lab_pixels, mean, std
    else:  # RGB
        mean = np.mean(pixels, axis=0)
        std = np.std(pixels, axis=0)
        return pixels, mean, std

def visualize_color_distribution(pixels, color_space="RGB", output_dir="."):
    """
    Visualize the distribution of color values in the specified color space.
    
    Args:
        pixels: Array of pixel values in the specified color space
        color_space: Color space name for plot titles
        output_dir: Directory to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    channel_names = ["R", "G", "B"] if color_space == "RGB" else ["L", "a", "b"]
    
    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        ax.hist(pixels[:, i], bins=50, alpha=0.7)
        ax.set_title(f'{color_space} {name} Channel Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{color_space}_distribution.png"))
    plt.close()

def segment_pumpkins(image, rgb_mean, rgb_std, lab_mean, lab_std):
    """
    Segment the image to isolate pumpkins based on color information.
    
    Args:
        image: Input image in BGR format
        rgb_mean: Mean RGB values of pumpkin pixels
        rgb_std: Standard deviation of RGB values of pumpkin pixels
        lab_mean: Mean LAB values of pumpkin pixels
        lab_std: Standard deviation of LAB values of pumpkin pixels
        
    Returns:
        Binary segmentation mask with white objects (pumpkins) on black background
    """
    # Convert image to RGB and LAB color spaces
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab_image = color.rgb2lab(rgb_image / 255.0)
    
    # Create color-based masks
    # RGB mask
    lower_rgb = np.maximum(rgb_mean - 2.5 * rgb_std, 0)
    upper_rgb = np.minimum(rgb_mean + 2.5 * rgb_std, 255)
    
    rgb_mask = cv2.inRange(rgb_image, lower_rgb.astype(np.uint8), upper_rgb.astype(np.uint8))
    
    # LAB mask
    h, w, _ = lab_image.shape
    lab_image_flat = lab_image.reshape(-1, 3)
    
    # Compute Mahalanobis distance in LAB space
    lab_cov = np.diag(lab_std ** 2)
    lab_diff = lab_image_flat - lab_mean
    
    mahalanobis_dist = np.sqrt(np.sum((lab_diff @ np.linalg.inv(lab_cov)) * lab_diff, axis=1))
    mahalanobis_dist = mahalanobis_dist.reshape(h, w)
    
    lab_mask = (mahalanobis_dist < 3.0).astype(np.uint8) * 255
    
    # Combine masks
    combined_mask = cv2.bitwise_and(rgb_mask, lab_mask)
    
    # Post-processing to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    
    return mask_closed

def main():
    # Paths to the annotated and original images (modify these paths as needed)
    annotated_image_path = "annotated/EB-02-660_0595_0341marked.JPG"
    original_image_path = "annotated/EB-02-660_0595_0341.JPG"
    output_dir = "./pumpkin_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the images
    annotated_image = cv2.imread(annotated_image_path)
    original_image = cv2.imread(original_image_path)
    
    if annotated_image is None:
        print(f"Could not read annotated image: {annotated_image_path}")
        return
    
    if original_image is None:
        print(f"Could not read original image: {original_image_path}")
        return
    
    # Extract the annotation mask
    annotation_mask = extract_annotation_mask(annotated_image)
    
    if cv2.countNonZero(annotation_mask) == 0:
        print("No red annotations found in the annotated image. Please check the annotation color.")
        return
    
    print(f"Found {cv2.countNonZero(annotation_mask)} annotated pixels.")
    
    # Extract the actual pumpkin pixels from the original image using the annotation mask
    pumpkin_pixels = extract_pumpkin_pixels_using_mask(original_image, annotation_mask)
    
    if len(pumpkin_pixels) == 0:
        print("No valid pumpkin pixels found using the annotation mask.")
        return
    
    print(f"Analyzing {len(pumpkin_pixels)} pumpkin pixels from the original image.")
    
    # Compute color statistics in RGB space
    rgb_pixels, rgb_mean, rgb_std = compute_color_statistics(pumpkin_pixels, "RGB")
    print(f"RGB Mean: {rgb_mean}")
    print(f"RGB Std: {rgb_std}")
    
    # Compute color statistics in CieLAB space
    lab_pixels, lab_mean, lab_std = compute_color_statistics(pumpkin_pixels, "CieLAB")
    print(f"CieLAB Mean: {lab_mean}")
    print(f"CieLAB Std: {lab_std}")
    
    # Visualize color distributions
    visualize_color_distribution(rgb_pixels, "RGB", output_dir)
    visualize_color_distribution(lab_pixels, "CieLAB", output_dir)
    
    # Segment the original image using the color statistics
    segmentation_mask = segment_pumpkins(original_image, rgb_mean, rgb_std, lab_mean, lab_std)
    
    # Save the annotation mask and segmentation result
    cv2.imwrite(os.path.join(output_dir, "annotation_mask.png"), annotation_mask)
    cv2.imwrite(os.path.join(output_dir, "segmentation_mask.png"), segmentation_mask)
    
    # Create binary output (white objects on black background)
    binary_output = np.zeros_like(original_image)
    binary_output[segmentation_mask > 0] = [255, 255, 255]
    cv2.imwrite(os.path.join(output_dir, "binary_result.png"), binary_output)
    
    # Display results in a figure
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title('Annotated Image')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original (Non-annotated) Image')
    
    plt.subplot(2, 2, 3)
    plt.imshow(annotation_mask, cmap='gray')
    plt.title('Annotation Mask')
    
    plt.subplot(2, 2, 4)
    plt.imshow(segmentation_mask, cmap='gray')
    plt.title('Segmentation Result')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "results_summary.png"))
    plt.close()
    
    # Save the color statistics to a file
    with open(os.path.join(output_dir, "color_statistics.txt"), "w") as f:
        f.write(f"RGB Mean: {rgb_mean}\n")
        f.write(f"RGB Std: {rgb_std}\n")
        f.write(f"CieLAB Mean: {lab_mean}\n")
        f.write(f"CieLAB Std: {lab_std}\n")
    
    print(f"Processing complete. Results saved to {output_dir}.")

if __name__ == "__main__":
    main()