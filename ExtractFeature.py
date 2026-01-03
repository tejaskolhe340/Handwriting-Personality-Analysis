"""
Handwriting Analysis Feature Extraction
This script extracts graphological features from handwriting samples for personality prediction.
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
from typing import List, Tuple, Dict, Any

# Suppress warnings
warnings.simplefilter('ignore')

# Constants
ANCHOR_POINT = 6000
MIDZONE_THRESHOLD = 15000

class HandwritingFeatures:
    """Container for extracted handwriting features"""
    def __init__(self):
        self.baseline_angle = 0.0
        self.top_margin = 0.0
        self.letter_size = 0.0
        self.line_spacing = 0.0
        self.word_spacing = 0.0
        self.pen_pressure = 0.0
        self.slant_angle = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert features to dictionary"""
        return {
            'baseline_angle': self.baseline_angle,
            'top_margin': self.top_margin,
            'letter_size': self.letter_size,
            'line_spacing': self.line_spacing,
            'word_spacing': self.word_spacing,
            'pen_pressure': self.pen_pressure,
            'slant_angle': self.slant_angle
        }

def bilateral_filter(image: np.ndarray, d: int) -> np.ndarray:
    """Apply bilateral filtering to image"""
    return cv2.bilateralFilter(image, d, 50, 50)

def median_filter(image: np.ndarray, d: int) -> np.ndarray:
    """Apply median filtering to image"""
    return cv2.medianBlur(image, d)

def threshold_image(image: np.ndarray, t: int) -> np.ndarray:
    """Convert to grayscale and apply inverted binary threshold"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
    return thresh

def dilate_image(image: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
    """Dilate image with given kernel size"""
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def erode_image(image: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
    """Erode image with given kernel size"""
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def straighten_image(image: np.ndarray, features: HandwritingFeatures) -> np.ndarray:
    """
    Straighten contours in the image horizontally and calculate baseline angle.
    Returns straightened image and updates features.baseline_angle.
    """
    filtered = bilateral_filter(image, 3)
    thresh = threshold_image(filtered, 120)
    dilated = dilate_image(thresh, (5, 100))
    
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    angle_sum = 0.0
    contour_count = 0
    
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        
        # Skip non-line contours
        if h > w or h < 20:
            continue
            
        roi = image[y:y+h, x:x+w]
        
        # Skip small lines
        if w < image.shape[1]/2:
            image[y:y+h, x:x+w] = 255
            continue
            
        # Straighten the contour
        rect = cv2.minAreaRect(ctr)
        angle = rect[2]
        if angle < -45.0:
            angle += 90.0
            
        rot = cv2.getRotationMatrix2D(((x+w)/2, (y+h)/2), angle, 1)
        extract = cv2.warpAffine(roi, rot, (w, h), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(255, 255, 255))
        image[y:y+h, x:x+w] = extract
        
        angle_sum += angle
        contour_count += 1
    
    # Calculate average baseline angle
    features.baseline_angle = angle_sum / contour_count if contour_count > 0 else 0.0
    print(f"Average baseline angle: {features.baseline_angle:.2f}°")
    
    return image

def horizontal_projection(img: np.ndarray) -> List[int]:
    """Calculate horizontal projection of image"""
    return [np.sum(row) for row in img]

def vertical_projection(img: np.ndarray) -> List[int]:
    """Calculate vertical projection of image"""
    return [np.sum(col) for col in img.T]

def extract_lines(img: np.ndarray, features: HandwritingFeatures) -> List[Tuple[int, int]]:
    """
    Extract lines of handwritten text using horizontal projection.
    Updates letter_size, line_spacing, and top_margin features.
    Returns list of (start, end) line indices.
    """
    filtered = bilateral_filter(img, 5)
    thresh = threshold_image(filtered, 160)
    hp_list = horizontal_projection(thresh)
    
    # Calculate top margin
    top_margin_count = 0
    for row_sum in hp_list:
        if row_sum <= 255:
            top_margin_count += 1
        else:
            break
    
    # Extract lines
    lines = []
    line_top = space_top = 0
    set_line_top = set_space_top = True
    space_zero = []
    
    for i, row_sum in enumerate(hp_list):
        if row_sum == 0:  # Blank space
            if set_space_top:
                space_top = i
                set_space_top = False
            if i < len(hp_list)-1 and hp_list[i+1] == 0:
                continue
            space_zero.append(i - space_top)
            set_space_top = True
            
        elif row_sum > 0:  # Text line
            if set_line_top:
                line_top = i
                set_line_top = False
            if i < len(hp_list)-1 and hp_list[i+1] > 0:
                continue
            if i - line_top < 20:  # Skip small lines
                set_line_top = True
                continue
            lines.append((line_top, i))
            set_line_top = True
    
    # Calculate features
    space_nonzero = midzone = lines_with_midzone = 0
    for start, end in lines:
        segment = hp_list[start:end]
        space_nonzero += sum(1 for s in segment if s < MIDZONE_THRESHOLD)
        midzone += sum(1 for s in segment if s >= MIDZONE_THRESHOLD)
        if any(s >= MIDZONE_THRESHOLD for s in segment):
            lines_with_midzone += 1
    
    lines_with_midzone = max(lines_with_midzone, 1)  # Prevent division by zero
    
    avg_letter_size = midzone / lines_with_midzone
    avg_line_spacing = (space_nonzero + sum(space_zero[1:-1])) / lines_with_midzone
    
    features.letter_size = avg_letter_size
    features.line_spacing = avg_line_spacing / avg_letter_size if avg_letter_size else 0
    features.top_margin = top_margin_count / avg_letter_size if avg_letter_size else 0
    
    print(f"Letter size: {avg_letter_size:.2f}px")
    print(f"Line spacing ratio: {features.line_spacing:.2f}")
    print(f"Top margin ratio: {features.top_margin:.2f}")
    
    return lines

def extract_words(image: np.ndarray, lines: List[Tuple[int, int]], features: HandwritingFeatures) -> List[Tuple[int, int, int, int]]:
    """
    Extract words from lines using vertical projection.
    Updates word_spacing feature.
    Returns list of (y1, y2, x1, x2) word coordinates.
    """
    filtered = bilateral_filter(image, 5)
    thresh = threshold_image(filtered, 180)
    width = thresh.shape[1]
    
    words = []
    space_zero = []
    
    for line_start, line_end in lines:
        line_img = thresh[line_start:line_end, 0:width]
        vp = vertical_projection(line_img)
        
        word_start = space_start = 0
        set_word_start = set_space_start = True
        spaces = []
        
        for j, col_sum in enumerate(vp):
            if col_sum == 0:  # Space
                if set_space_start:
                    space_start = j
                    set_space_start = False
                if j < len(vp)-1 and vp[j+1] == 0:
                    continue
                space_width = j - space_start
                if space_width > features.letter_size/2:
                    spaces.append(space_width)
                set_space_start = True
                
            elif col_sum > 0:  # Word
                if set_word_start:
                    word_start = j
                    set_word_start = False
                if j < len(vp)-1 and vp[j+1] > 0:
                    continue
                
                # Check word height
                word_height = sum(1 for k in range(line_start, line_end) 
                                if np.sum(thresh[k, word_start:j+1]))
                if word_height > features.letter_size/2:
                    words.append((line_start, line_end, word_start, j+1))
                set_word_start = True
        
        space_zero.extend(spaces[1:-1])
    
    # Calculate word spacing
    if space_zero:
        avg_word_spacing = sum(space_zero) / len(space_zero)
        features.word_spacing = avg_word_spacing / features.letter_size if features.letter_size else 0
        print(f"Word spacing ratio: {features.word_spacing:.2f}")
    else:
        features.word_spacing = 0
    
    return words

def extract_slant_angle(img: np.ndarray, words: List[Tuple[int, int, int, int]], features: HandwritingFeatures) -> None:
    """Determine average slant angle of handwriting"""
    theta = [-0.785398, -0.523599, -0.261799, -0.0872665,
             0.01, 0.0872665, 0.261799, 0.523599, 0.785398]  # -45° to +45°
    
    filtered = bilateral_filter(img, 5)
    thresh = threshold_image(filtered, 180)
    
    s_function = np.zeros(9)
    counts = np.zeros(9)
    
    for i, angle in enumerate(theta):
        s_temp = 0.0
        count = 0
        
        for y1, y2, x1, x2 in words:
            word_img = thresh[y1:y2, x1:x2]
            height, width = word_img.shape
            shift = (math.tan(angle) * height) / 2
            pad = abs(int(shift))
            
            # Create padded image
            padded = np.zeros((height, width + pad*2), dtype=np.uint8)
            padded[:, pad:width+pad] = word_img
            
            # Apply slant transformation
            pts1 = np.float32([[width/2, 0], [width/4, height], [3*width/4, height]])
            pts2 = np.float32([[width/2+shift, 0], 
                             [width/4-shift, height], 
                             [3*width/4-shift, height]])
            M = cv2.getAffineTransform(pts1, pts2)
            deslanted = cv2.warpAffine(padded, M, (width, height))
            
            # Analyze vertical projection
            vp = vertical_projection(deslanted)
            for col_sum in vp:
                if col_sum == 0:
                    continue
                    
                num_pixels = col_sum // 255
                if num_pixels < height/3:
                    continue
                    
                # Calculate column metrics
                if col_sum >= deslanted.shape[1]:
                    col_sum = deslanted.shape[1] - 1  # Prevent out-of-bo
                column = deslanted[:, col_sum]
                top = np.argmax(column > 0)
                bottom = height - np.argmax(column[::-1] > 0)
                delta_y = height - (top + bottom)
                
                if delta_y > 0:
                    h_sq = (num_pixels/delta_y)**2
                    s_temp += (h_sq * num_pixels) / height
                    count += 1
        
        s_function[i] = s_temp
        counts[i] = count
    
    # Determine dominant slant angle
    max_idx = np.argmax(s_function)
    angle_map = {
        0: 45, 1: 30, 2: 15, 3: 5,
        4: 0,  # No slant
        5: -5, 6: -15, 7: -30, 8: -45
    }
    
    # Handle ambiguous cases
    if max_idx == 4:  # No slant case
        p = s_function[4] / s_function[3]
        q = s_function[4] / s_function[5]
        if not ((p <= 1.2 and q <= 1.2) or (p > 1.4 and q > 1.4)):
            max_idx = 9  # Irregular slant
    
    features.slant_angle = angle_map.get(max_idx, 180)  # 180 for irregular
    print(f"Slant angle: {features.slant_angle}°")

def extract_pen_pressure(image: np.ndarray, features: HandwritingFeatures) -> None:
    """Calculate average pen pressure from handwriting"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    filtered = bilateral_filter(inverted, 3)
    _, thresh = cv2.threshold(filtered, 100, 255, cv2.THRESH_TOZERO)
    
    # Calculate average intensity of non-zero pixels
    non_zero = thresh[thresh > 0]
    features.pen_pressure = np.mean(non_zero) if non_zero.size > 0 else 0
    print(f"Pen pressure: {features.pen_pressure:.2f}")

def analyze_handwriting(image_path: str, debug: bool = False) -> HandwritingFeatures:
    """
    Main function to analyze handwriting image
    Returns extracted features
    """
    features = HandwritingFeatures()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    if debug:
        cv2.imshow("Original", image)
    
    # Extract features
    straightened = straighten_image(image.copy(), features)
    lines = extract_lines(straightened.copy(), features)
    words = extract_words(straightened.copy(), lines, features)
    extract_slant_angle(straightened.copy(), words, features)
    extract_pen_pressure(image.copy(), features)
    
    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return features

if __name__ == "__main__":
    # Example usage
    try:
        features = analyze_handwriting("images/007-0.png", debug=True)
        print("\nExtracted Features:")
        for name, value in features.to_dict().items():
            print(f"{name.replace('_', ' ').title():<20}: {value:.2f}")
    except Exception as e:
        print(f"Error: {e}")