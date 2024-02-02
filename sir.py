



# my modified code


# from PIL import Image, ImageDraw, ImageOps, ImageEnhance
# import pytesseract
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def enhance_image_quality(img):
#     # Increase the contrast of the image
#     alpha = 1.5
#     beta = 30
#     enhanced_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

#     # Increase the sharpness of the image using GaussianBlur
#     enhanced_img = cv2.GaussianBlur(enhanced_img, (0, 0), 3)
#     enhanced_img = cv2.addWeighted(img, 1.5, enhanced_img, -0.5, 0)

#     return enhanced_img

# def convert_to_portrait(img):
#     # Rotate the image 90 degrees counterclockwise
#     rotated_img = cv2.transpose(img)
#     rotated_img = cv2.flip(rotated_img, 1)
#     return rotated_img

# def extract_text_from_section(img_path):
#     img = cv2.imread(img_path)

#     if img is None:
#         print("Error: Unable to load the image. Please check the path you have entered.")
#         return

#     # Check if the image is in landscape mode and convert it to portrait
#     if img.shape[1] > img.shape[0]:
#         img = convert_to_portrait(img)

#     # Enhance the image quality
#     img = enhance_image_quality(img)
    
#     # Create a figure and axis
#     fig, ax = plt.subplots(1, figsize=(10, 10))

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.bilateralFilter(gray, 9, 10, 10)
#     binary_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#     kernel = np.ones((3, 3), np.uint8)
#     fill_space = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(fill_space, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     min_length = 0.75 * img.shape[1]
#     min_aspect_ratio = 3

#     # List to store the coordinates of highlighted lines
#     highlighted_lines = []

#     # Iterate through the contours and check line width
#     for i in contours:
#         epsilon = 0.02 * cv2.arcLength(i, True)
#         approx = cv2.approxPolyDP(i, epsilon, True)

#         x, y, w, h = cv2.boundingRect(approx)
#         aspect_ratio = float(w) / h
#         if w >= min_length and aspect_ratio >= min_aspect_ratio:
#             # Store the y-coordinate of the highlighted line
#             highlighted_lines.append(y + h // 2)

#             # Draw a red circle at the starting point
#             cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

#             # Draw a red circle at the ending point
#             cv2.circle(img, (x + w, y + h), 5, (0, 0, 255), -1)

#             # Draw a horizontal line passing through the starting and ending points
#             cv2.line(img, (0, y + h//2), (img.shape[1], y + h//2), (0, 255, 0), 2)

#     # Sort the highlighted lines in ascending order
#     highlighted_lines.sort()

#     # Create sections based on highlighted lines
#     sections = []
#     start_y = 0

#     for line_y in highlighted_lines:
#         # Extract a section from start_y to line_y
#         section = img[start_y:line_y, :]
#         sections.append(section)

#         # Draw a rectangle around the section
#         rect = plt.Rectangle((0, start_y), img.shape[1], line_y - start_y, edgecolor='blue', linewidth=2, facecolor='none')
#         ax.add_patch(rect)

#         # Update start_y for the next section
#         start_y = line_y

#     # Check if there is a section from the last highlighted line to the bottom of the image
#     if start_y < img.shape[0]:
#         final_section = img[start_y:, :]
#         sections.append(final_section)

#         # Draw a rectangle around the final section
#         rect = plt.Rectangle((0, start_y), img.shape[1], img.shape[0] - start_y, edgecolor='blue', linewidth=2, facecolor='none')
#         ax.add_patch(rect)

#     # Perform OCR on all sections and find the section with the most variables
#     most_variable_section = None
#     max_variable_count = 0

#     for i, section in enumerate(sections):
#         section_image = Image.fromarray(cv2.cvtColor(section, cv2.COLOR_BGR2RGB))
#         section_text = pytesseract.image_to_string(section_image, config='--oem 3 --psm 6')

#         # Count the number of variables (you can adjust this logic based on your specific use case)
#         variable_count = len(section_text.split())
#         # print(f"Section {i+1} Variable Count: {variable_count}")

#         if variable_count > max_variable_count:
#             max_variable_count = variable_count
#             most_variable_section = section_text

#     # Print the text of the section with the most variables
#     print("\nText of Section with Most Variables:")
#     print(most_variable_section)

#     # # data_dictionary = {}
#     # # lines = most_variable_section.split('\n')
#     # # for line in lines:
#     # #     words = line.split()
#     # #     if words:
#     # #         heading = words[0]
#     # #         value = ' '.join(words[1:])
#     # #         data_dictionary[heading] = value

#     # print("\nData of Section with Most Variables:")
#     # print(data_dictionary)
    
#     data_tuples = []
#     lines = most_variable_section.split('\n')
#     for line in lines:
#         words = line.split()
#         if words:
#             heading = words[0]
#             value = ' '.join(words[1:])
#             data_tuples.append((heading, value))

#     print("\nData of Section with Most Variables:")
#     print(data_tuples)

#     plt.imshow(img)
#     plt.show()
# # Example usage
# img_path = 'images/ocr17.jpg'
# extract_text_from_section(img_path)
    

# # Convolution Filtering
import cv2
import pytesseract
from matplotlib import pyplot as plt

# Set the path to the Tesseract executable (update the path accordingly)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to perform convolution filtering on an image
def apply_convolution(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply convolution filtering (e.g., Gaussian blur)
    blurred_image = cv2.GaussianBlur(gray_image, (1, 5), 0)

    return blurred_image

# Function to perform OCR on an image
def perform_ocr(image_path):
    # Apply convolution filtering
    filtered_image = apply_convolution(image_path)

    # Use pytesseract to perform OCR
    text = pytesseract.image_to_string(filtered_image)

    return text, filtered_image

# Path to the image file
image_path = 'pathReport.jpeg'

# Perform OCR and get the extracted text and enhanced image
extracted_text, enhanced_image = perform_ocr(image_path)

# Display the original and enhanced images
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Image')
plt.axis('off')

plt.show()

# Print the extracted text
print("Extracted Text:")
print(extracted_text)


##----------------------------

# data sequencing error
# import cv2
# import pytesseract
# from matplotlib import pyplot as plt

# # Set the path to the Tesseract executable (update the path accordingly)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Function to perform median filtering on an image
# def apply_median_filter(image_path):
#     # Read the image
#     image = cv2.imread(image_path)

#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply median filtering
#     filtered_image = cv2.medianBlur(gray_image, 1)  # Change the kernel size (e.g., 5) as needed

#     return filtered_image

# # Function to perform OCR on an image
# def perform_ocr(image_path, custom_config):
#     # Apply median filtering
#     filtered_image = apply_median_filter(image_path)

#     # Save the modified image
#     output_image_path = 'median_filter.jpeg'
#     cv2.imwrite(output_image_path, filtered_image)

#     # Use pytesseract to perform OCR with custom configuration
#     text = pytesseract.image_to_string(filtered_image, config=custom_config)

#     return text, output_image_path

# # Path to the image file
# image_path = 'pathReport.jpeg'
# custom_config = '--oem 2 --psm 6'

# # Perform OCR and get the extracted text and enhanced image
# extracted_text, output_image_path = perform_ocr(image_path, custom_config)

# # Display the original and enhanced images
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
# plt.title('Enhanced Image (Median Filter)')
# plt.axis('off')

# plt.show()

# # Print the extracted text
# print("Extracted Text:")
# print(extracted_text) 
#-----------------------------------
import cv2
import pytesseract
from matplotlib import pyplot as plt

# Function to perform Canny Edge Detection on an image with resolution enhancement and darker text
def enhance_resolution_and_darken_text_canny(image_path, low_threshold, high_threshold, scale_factor, contrast_factor):
    # Read the image and enhance resolution
    image = cv2.imread(image_path)
    high_res_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale and adjust contrast
    gray_image = cv2.cvtColor(high_res_image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.multiply(gray_image, contrast_factor)

    # Apply Canny Edge Detection
    edges = cv2.Canny(enhanced_image, low_threshold, high_threshold)

    return edges

# Function to perform OCR on an image
def perform_ocr_canny(image_path):
    # Apply Canny Edge Detection with resolution enhancement and darker text
    edges = enhance_resolution_and_darken_text_canny(image_path, 50, 150, 2.0, 1.2)

    # Save the modified image
    output_image_path = 'canny_edge_detection_enhanced.jpeg'
    cv2.imwrite(output_image_path, edges)

    # Use pytesseract to perform OCR
    text = pytesseract.image_to_string(edges)

    return text, output_image_path

# Path to the image file
image_path = 'pathReport.jpeg'

# Perform OCR and get the extracted text and enhanced image
extracted_text, output_image_path = perform_ocr_canny(image_path)

# Display the original and enhanced images
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
plt.title('Enhanced Image (Canny Edge Detection)')
plt.axis('off')

plt.show()

# Print the extracted text
print("Extracted Text:")
print(extracted_text)


#------------------------------------

# #cannot use not good show blank page
# import cv2
# import pytesseract
# from matplotlib import pyplot as plt

# # Function to perform Sobel or Prewitt filtering on an image with resolution enhancement and darker text
# def enhance_resolution_and_darken_text_sobel_prewitt(image_path, filter_type, scale_factor, contrast_factor):
#     # Read the image and enhance resolution
#     image = cv2.imread(image_path)
#     high_res_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

#     # Convert to grayscale and adjust contrast
#     gray_image = cv2.cvtColor(high_res_image, cv2.COLOR_BGR2GRAY)
#     enhanced_image = cv2.multiply(gray_image, contrast_factor)

#     # Apply Sobel or Prewitt filter
#     if filter_type == 'sobel':
#         edges = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 1, ksize=3)
#     elif filter_type == 'prewitt':
#         kernel_x = cv2.getDerivKernels(1, 0, 3, normalize=True)
#         kernel_y = cv2.getDerivKernels(0, 1, 3, normalize=True)
#         edges_x = cv2.filter2D(enhanced_image, cv2.CV_64F, kernel_x)
#         edges_y = cv2.filter2D(enhanced_image, cv2.CV_64F, kernel_y)
#         edges = cv2.magnitude(edges_x, edges_y)

#     # Normalize and convert to uint8 for display
#     edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

#     return edges

# # Function to perform OCR on an image
# def perform_ocr_sobel_prewitt(image_path):
#     # Apply Sobel filter with resolution enhancement and darker text
#     edges = enhance_resolution_and_darken_text_sobel_prewitt(image_path, 'sobel', 2.0, 1.2)

#     # Save the modified image
#     output_image_path = 'sobel_edge_detection_enhanced.jpeg'
#     cv2.imwrite(output_image_path, edges)

#     # Use pytesseract to perform OCR
#     text = pytesseract.image_to_string(edges)

#     return text, output_image_path

# # Path to the image file
# image_path = 'pathReport.jpeg'

# # Perform OCR and get the extracted text and enhanced image
# extracted_text, output_image_path = perform_ocr_sobel_prewitt(image_path)

# # Display the original and enhanced images
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
# plt.title('Enhanced Image (Sobel Edge Detection)')
# plt.axis('off')

# plt.show()

# # Print the extracted text
# print("Extracted Text:")
# # print(extracted_text)

#-----------------------
# import cv2
# import pytesseract
# from matplotlib import pyplot as plt

# # Function to perform Bilateral Filtering on an image with resolution enhancement and darker text
# def enhance_resolution_and_darken_text_bilateral(image_path, diameter, sigma_color, sigma_space, scale_factor, contrast_factor):
#     # Read the image and enhance resolution
#     image = cv2.imread(image_path)
#     high_res_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

#     # Adjust contrast
#     enhanced_image = cv2.multiply(high_res_image, contrast_factor)

#     # Apply Bilateral Filtering
#     filtered_image = cv2.bilateralFilter(enhanced_image, diameter, sigma_color, sigma_space)

#     return filtered_image

# # Function to perform OCR on an image
# def perform_ocr_bilateral(image_path):
#     # Apply Bilateral Filtering with resolution enhancement and darker text
#     filtered_image = enhance_resolution_and_darken_text_bilateral(image_path, diameter=9, sigma_color=75, sigma_space=75, scale_factor=2.0, contrast_factor=1.2)

#     # Save the modified image
#     output_image_path = 'bilateral_filtering_enhanced.jpeg'
#     cv2.imwrite(output_image_path, filtered_image)

#     # Use pytesseract to perform OCR
#     text = pytesseract.image_to_string(filtered_image)

#     return text, output_image_path

# # Path to the image file
# image_path = 'pathReport.jpeg'

# # Perform OCR and get the extracted text and enhanced image
# extracted_text, output_image_path = perform_ocr_bilateral(image_path)

# # Display the original and enhanced images
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(cv2.imread(output_image_path), cv2.COLOR_BGR2RGB))
# plt.title('Enhanced Image (Bilateral Filtering)')
# plt.axis('off')

# plt.show()

# # Print the extracted text
# print("Extracted Text:")
# print(extracted_text)
#--------------------
import cv2
import pytesseract
from matplotlib import pyplot as plt

# Function to perform morphological operations on an image
def apply_morphological_operations(image_path, scale_factor, contrast_factor):
    # Read the image and enhance resolution
    image = cv2.imread(image_path)
    high_res_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale and adjust contrast
    gray_image = cv2.cvtColor(high_res_image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.multiply(gray_image, contrast_factor)

    # Apply thresholding (adjust threshold value as needed)
    _, thresholded_image = cv2.threshold(enhanced_image, 128, 255, cv2.THRESH_BINARY)

    # Apply morphological operations (e.g., dilation followed by erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_image = cv2.dilate(thresholded_image, kernel, iterations=1)
    morph_image = cv2.erode(morph_image, kernel, iterations=1)

    return morph_image

# Function to perform OCR on an image
def perform_ocr_morph(image_path):
    # Apply morphological operations with resolution enhancement and darker text
    morph_image = apply_morphological_operations(image_path, scale_factor=2.0, contrast_factor=0.6)

    # Save the modified image
    output_image_path = 'morphological_operations_enhanced.jpeg'
    cv2.imwrite(output_image_path, morph_image)

    # Use pytesseract to perform OCR
    text = pytesseract.image_to_string(morph_image)

    return text, output_image_path

# Path to the image file
image_path = 'pathReport.jpeg'

# Perform OCR and get the extracted text and enhanced image
extracted_text, output_image_path = perform_ocr_morph(image_path)

# Display the original and enhanced images
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(cv2.imread(output_image_path), cv2.COLOR_BGR2RGB))
plt.title('Enhanced Image (Morphological Operations)')
plt.axis('off')

plt.show()

# Print the extracted text
print("Extracted Text:")
print(extracted_text)

