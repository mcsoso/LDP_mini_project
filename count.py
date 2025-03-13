import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color
import os

def segment_pumpkins(image, rgb_mean, rgb_std, lab_mean, lab_std):
    """
    Segment the image to isolate pumpkins based on learned color information.
    
    Args:
        image: Input image in BGR format
        rgb_mean: Mean RGB values of pumpkin pixels (from training)
        rgb_std: Standard deviation of RGB values (from training)
        lab_mean: Mean LAB values of pumpkin pixels (from training)
        lab_std: Standard deviation of LAB values (from training)
        
    Returns:
        Binary segmentation mask with white objects (pumpkins) on black background
    """
    # Convert image to RGB and LAB color spaces
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab_image = color.rgb2lab(rgb_image / 255.0)
    
    # Create color-based masks
    # RGB mask
    lower_rgb = np.maximum(rgb_mean - 4 * rgb_std, 0)
    upper_rgb = np.minimum(rgb_mean + 4 * rgb_std, 255)
    
    rgb_mask = cv2.inRange(rgb_image, lower_rgb.astype(np.uint8), upper_rgb.astype(np.uint8))
    
    # LAB mask
    h, w, _ = lab_image.shape
    lab_image_flat = lab_image.reshape(-1, 3)
    
    # Compute Mahalanobis distance in LAB space
    lab_cov = np.diag(lab_std ** 2)
    lab_diff = lab_image_flat - lab_mean
    
    mahalanobis_dist = np.sqrt(np.sum((lab_diff @ np.linalg.inv(lab_cov)) * lab_diff, axis=1))
    mahalanobis_dist = mahalanobis_dist.reshape(h, w)
    
    lab_mask = (mahalanobis_dist < 5.0).astype(np.uint8) * 255
    
    # Combine masks
    combined_mask = cv2.bitwise_and(rgb_mask, lab_mask)
    
    # Post-processing to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    
    return mask_closed

def main():
    # Get input paths
    stats_file_path = "./pumpkin_results/color_statistics.txt"
    #new_image_path = "pictures/EB-02-660_0595_0007.jpg"
    #new_image_path = "pictures/EB-02-660_0595_0413.jpg"
    new_image_path = "pictures/ortho_crop.png"
    output_dir = "./pumpkin_detection_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the color statistics from the training script output
    if not os.path.exists(stats_file_path):
        print(f"Color statistics file not found: {stats_file_path}")
        print("Please run the training script first.")
        return
    
    # Parse the color statistics file
    rgb_mean = None
    rgb_std = None
    lab_mean = None
    lab_std = None
    
    try:
        with open(stats_file_path, "r") as f:
            for line in f:
                if "RGB Mean" in line:
                    rgb_mean = np.array(eval(line.split(": ")[1]))
                elif "RGB Std" in line:
                    rgb_std = np.array(eval(line.split(": ")[1]))
                elif "CieLAB Mean" in line:
                    lab_mean = np.array(eval(line.split(": ")[1]))
                elif "CieLAB Std" in line:
                    lab_std = np.array(eval(line.split(": ")[1]))
    except Exception as e:
        print(f"Error parsing color statistics file: {e}")
        return
    
    # Check if any of the required statistics is None
    if rgb_mean is None or rgb_std is None or lab_mean is None or lab_std is None:
        print("Failed to load all required color statistics.")
        return
    
    print("Loaded color statistics from training:")
    print(f"RGB Mean: {rgb_mean}")
    print(f"RGB Std: {rgb_std}")
    print(f"CieLAB Mean: {lab_mean}")
    print(f"CieLAB Std: {lab_std}")
    
    # Read the new image
    image = cv2.imread(new_image_path)
    
    if image is None:
        print(f"Could not read image: {new_image_path}")
        return
    
    print(f"Processing image: {new_image_path}")
    
    # Segment the image using learned color model
    segmentation_mask = segment_pumpkins(image, rgb_mean, rgb_std, lab_mean, lab_std)
    
    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(new_image_path))[0]
    
    # Save the segmentation result
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_segmentation.png"), segmentation_mask)
    
    # Create segmented RGB image (pumpkins only)
    segmented_rgb = cv2.bitwise_and(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        mask=segmentation_mask
    )
    
    # Save segmented RGB image
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_segmented_rgb.png"), 
                cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR))
    
    # Create binary output (white objects on black background)
    binary_output = np.zeros_like(image)
    binary_output[segmentation_mask > 0] = [255, 255, 255]
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_binary.png"), binary_output)
    
    # Count the number of pumpkins
    # First convert the mask to binary
    binary_mask = (segmentation_mask > 0).astype(np.uint8)
    # Apply connected components analysis to count objects
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    # Subtract 1 from num_labels because label 0 is the background
    pumpkin_count = num_labels - 1
    
    # Filter small objects (noise) by area
    min_area = 2  # Minimum area threshold (adjust as needed)
    filtered_pumpkin_count = 0
    for i in range(1, num_labels):  # Start from 1 to skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_pumpkin_count += 1
    
    print(f"Total pumpkins detected: {pumpkin_count}")
    print(f"Filtered pumpkins detected (removing small objects): {filtered_pumpkin_count}")
    
    # Save count information to a text file
    with open(os.path.join(output_dir, f"{base_name}_count.txt"), "w") as f:
        f.write(f"Image: {new_image_path}\n")
        f.write(f"Total pumpkins detected: {pumpkin_count}\n")
        f.write(f"Filtered pumpkins (area >= {min_area} pixels): {filtered_pumpkin_count}\n")
    
    # Create a visualization with detected pumpkins labeled
    labeled_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    
    # Draw centroids and labels on filtered pumpkins
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            # Get centroid coordinates
            cx = int(centroids[i][0])
            cy = int(centroids[i][1])
            
            # Draw circle at centroid
            cv2.circle(labeled_image, (cx, cy), 5, (0, 255, 0), -1)
            
            # Add label with pumpkin number
            cv2.putText(labeled_image, f"{i}", (cx - 10, cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save labeled image
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_labeled.png"),
                cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR))
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(segmentation_mask, cmap='gray')
    plt.title('Segmentation Result')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(binary_output, cv2.COLOR_BGR2RGB))
    plt.title(f'Binary Result - {filtered_pumpkin_count} pumpkins detected')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_results.png"))
    plt.close()
    
    print(f"Processing complete. Results saved to {output_dir}.")

if __name__ == "__main__":
    main()