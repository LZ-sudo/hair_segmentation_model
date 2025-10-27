"""
Computer Vision module for hair length estimation.
Handles A4 paper detection, calibration, and hair length measurement.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class ReferenceObject:
    """Detected reference object (A4 paper) for calibration."""
    contour: np.ndarray
    width_pixels: float
    height_pixels: float
    known_width_cm: float
    known_height_cm: float
    pixels_per_cm: float
    confidence: float
    center: Tuple[int, int]


@dataclass
class HairMeasurement:
    """Hair length measurement result."""
    top_point: Tuple[int, int]
    bottom_point: Tuple[int, int]
    length_pixels: float
    length_cm: float
    pixels_per_cm: float
    confidence: float


class HairLengthCV:
    """Computer Vision operations for hair length measurement."""
    
    # A4 paper dimensions (standard)
    A4_WIDTH_CM = 21.0
    A4_HEIGHT_CM = 29.7
    A4_ASPECT_RATIO = A4_HEIGHT_CM / A4_WIDTH_CM  # ~1.414
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CV module.
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        if config is None:
            config = {}
        
        # Detection method settings
        self.detection_method = config.get('detection_method', 'color')  # 'color', 'edges', or 'auto'
        self.paper_color = config.get('paper_color', 'yellow')  # Color of the A4 paper
        
        # A4 detection parameters
        self.edge_threshold1 = config.get('edge_threshold1', 50)
        self.edge_threshold2 = config.get('edge_threshold2', 150)
        self.min_area = config.get('min_area_pixels', 10000)
        self.aspect_ratio_tolerance = config.get('aspect_ratio_tolerance', 0.15)
        self.min_confidence = config.get('min_confidence', 0.7)
        
        # Morphological operation parameters
        self.morph_kernel_size = config.get('morph_kernel_size', 5)
        
        print("HairLengthCV initialized")
        print(f"  Detection method: {self.detection_method}")
        print(f"  Paper color: {self.paper_color}")
        print(f"  Edge thresholds: {self.edge_threshold1}, {self.edge_threshold2}")
        print(f"  Min area: {self.min_area} pixels")
        print(f"  Aspect ratio tolerance: {self.aspect_ratio_tolerance}")
    
    # ===== A4 PAPER DETECTION =====
    
    def detect_a4_paper_by_color(
        self, 
        image: np.ndarray, 
        color: str = 'yellow'
    ) -> Optional[ReferenceObject]:
        """
        Detect A4 paper by color (more robust for colored paper on colored walls).
        
        Args:
            image: Input image (BGR format)
            color: Color of the paper ('yellow', 'white', 'blue', 'green', 'red')
            
        Returns:
            ReferenceObject if A4 paper detected, None otherwise
        """
        print(f"\n[1/3] Detecting {color} A4 paper using color segmentation...")
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV
        color_ranges = {
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'green': ([40, 100, 100], [80, 255, 255]),
            'red': ([0, 100, 100], [10, 255, 255]),  # Red wraps around in HSV
            'orange': ([10, 100, 100], [20, 255, 255]),
            'pink': ([140, 100, 100], [170, 255, 255])
        }
        
        if color not in color_ranges:
            print(f"  ⚠ Unknown color: {color}, using edge detection instead")
            return self.detect_a4_paper_by_edges(image)
        
        lower, upper = color_ranges[color]
        lower = np.array(lower)
        upper = np.array(upper)
        
        # Create mask for the specified color
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  Found {len(contours)} {color} regions")
        
        if len(contours) == 0:
            print(f"  ✗ No {color} regions found")
            return None
        
        # Filter and score candidates
        candidates = []
        
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Prefer quadrilaterals but accept other shapes with good aspect ratio
            if len(approx) < 4:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)
            
            # Calculate aspect ratio
            aspect_ratio = max(h, w) / min(h, w)
            
            # Check if aspect ratio matches A4 (within tolerance)
            ratio_diff = abs(aspect_ratio - self.A4_ASPECT_RATIO)
            if ratio_diff > self.aspect_ratio_tolerance * self.A4_ASPECT_RATIO:
                continue
            
            # Calculate confidence based on:
            # 1. How close to ideal A4 ratio
            # 2. How rectangular the contour is (convexity)
            ratio_confidence = 1.0 - (ratio_diff / (self.aspect_ratio_tolerance * self.A4_ASPECT_RATIO))
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            rectangularity = area / hull_area if hull_area > 0 else 0
            
            confidence = ratio_confidence * 0.7 + rectangularity * 0.3
            
            # Add to candidates
            candidates.append({
                'contour': approx,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'confidence': confidence
            })
        
        if not candidates:
            print(f"  ✗ No {color} A4-shaped rectangles found")
            return None
        
        # Select best candidate (highest confidence, then largest area)
        best = max(candidates, key=lambda x: (x['confidence'], x['area']))
        
        print(f"  ✓ {color.capitalize()} A4 paper detected!")
        print(f"    Size: {best['width']}×{best['height']} pixels")
        print(f"    Aspect ratio: {best['aspect_ratio']:.3f} (expected: {self.A4_ASPECT_RATIO:.3f})")
        print(f"    Confidence: {best['confidence']:.2%}")
        
        # Determine orientation and known dimensions
        if best['height'] > best['width']:
            # Portrait orientation (standard)
            known_width_cm = self.A4_WIDTH_CM
            known_height_cm = self.A4_HEIGHT_CM
        else:
            # Landscape orientation
            known_width_cm = self.A4_HEIGHT_CM
            known_height_cm = self.A4_WIDTH_CM
        
        # Calculate pixels per cm (use height for vertical measurements)
        pixels_per_cm = best['height'] / known_height_cm
        
        print(f"    Calibration: {pixels_per_cm:.2f} pixels/cm")
        
        # Calculate center point
        center_x = best['x'] + best['width'] // 2
        center_y = best['y'] + best['height'] // 2
        
        return ReferenceObject(
            contour=best['contour'],
            width_pixels=best['width'],
            height_pixels=best['height'],
            known_width_cm=known_width_cm,
            known_height_cm=known_height_cm,
            pixels_per_cm=pixels_per_cm,
            confidence=best['confidence'],
            center=(center_x, center_y)
        )
    
    def detect_a4_paper_by_edges(self, image: np.ndarray) -> Optional[ReferenceObject]:
        """
        Detect A4 paper by edges (original method - works for any color paper).
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            ReferenceObject if A4 paper detected, None otherwise
        """
        print("\n[1/4] Detecting A4 paper using edge detection...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.edge_threshold1, self.edge_threshold2)
        
        # Morphological closing to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  Found {len(contours)} contours")
        
        if len(contours) == 0:
            print("  ✗ No contours found")
            return None
        
        # Filter and score candidates
        candidates = []
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Must be a quadrilateral (4 corners)
            if len(approx) != 4:
                continue
            
            # Calculate area
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)
            
            # Calculate aspect ratio
            aspect_ratio = max(h, w) / min(h, w)
            
            # Check if aspect ratio matches A4 (within tolerance)
            ratio_diff = abs(aspect_ratio - self.A4_ASPECT_RATIO)
            if ratio_diff > self.aspect_ratio_tolerance * self.A4_ASPECT_RATIO:
                continue
            
            # Calculate confidence score (based on how close to ideal A4 ratio)
            confidence = 1.0 - (ratio_diff / (self.aspect_ratio_tolerance * self.A4_ASPECT_RATIO))
            
            # Add to candidates
            candidates.append({
                'contour': approx,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'confidence': confidence
            })
        
        if not candidates:
            print("  ✗ No A4-shaped rectangles found")
            return None
        
        # Select best candidate (highest confidence, then largest area)
        best = max(candidates, key=lambda x: (x['confidence'], x['area']))
        
        print(f"  ✓ A4 paper detected!")
        print(f"    Size: {best['width']}×{best['height']} pixels")
        print(f"    Aspect ratio: {best['aspect_ratio']:.3f} (expected: {self.A4_ASPECT_RATIO:.3f})")
        print(f"    Confidence: {best['confidence']:.2%}")
        
        # Determine orientation and known dimensions
        if best['height'] > best['width']:
            # Portrait orientation (standard)
            known_width_cm = self.A4_WIDTH_CM
            known_height_cm = self.A4_HEIGHT_CM
        else:
            # Landscape orientation
            known_width_cm = self.A4_HEIGHT_CM
            known_height_cm = self.A4_WIDTH_CM
        
        # Calculate pixels per cm (use height for vertical measurements)
        pixels_per_cm = best['height'] / known_height_cm
        
        print(f"    Calibration: {pixels_per_cm:.2f} pixels/cm")
        
        # Calculate center point
        center_x = best['x'] + best['width'] // 2
        center_y = best['y'] + best['height'] // 2
        
        return ReferenceObject(
            contour=best['contour'],
            width_pixels=best['width'],
            height_pixels=best['height'],
            known_width_cm=known_width_cm,
            known_height_cm=known_height_cm,
            pixels_per_cm=pixels_per_cm,
            confidence=best['confidence'],
            center=(center_x, center_y)
        )
    
    def detect_a4_paper(self, image: np.ndarray, visualize: bool = False) -> Optional[ReferenceObject]:
        """
        Detect A4 paper in image using the configured detection method.
        
        Args:
            image: Input image (BGR format)
            visualize: If True, return visualization image
            
        Returns:
            ReferenceObject if A4 paper detected, None otherwise
        """
        # Try color-based detection first if configured
        detection_method = self.detection_method
        paper_color = self.paper_color
        
        if detection_method == 'color':
            result = self.detect_a4_paper_by_color(image, paper_color)
            if result is not None:
                return result
            # Fall back to edge detection
            print("  Color detection failed, trying edge detection...")
            return self.detect_a4_paper_by_edges(image)
        
        elif detection_method == 'edges':
            return self.detect_a4_paper_by_edges(image)
        
        else:
            # Try both methods
            print("\n[1/4] Trying multiple detection methods...")
            result = self.detect_a4_paper_by_color(image, paper_color)
            if result is not None:
                return result
            print("  Color detection failed, trying edge detection...")
            return self.detect_a4_paper_by_edges(image)
    
    # ===== HAIR MEASUREMENT =====
    
    def find_hair_extremes(self, hair_mask: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Find topmost and bottommost points of hair in the segmentation mask.
        
        Args:
            hair_mask: Binary hair mask (0 or 255)
            
        Returns:
            Tuple of (top_point, bottom_point) where each point is (x, y)
            Returns (None, None) if no hair found
        """
        print("\n[2/3] Finding hair extremes...")
        
        # Find all non-zero pixels (hair pixels)
        hair_pixels = np.where(hair_mask > 0)
        
        if len(hair_pixels[0]) == 0:
            print("  ✗ No hair pixels found")
            return None, None
        
        # Get topmost point (minimum y)
        top_idx = np.argmin(hair_pixels[0])
        top_y = hair_pixels[0][top_idx]
        top_x = hair_pixels[1][top_idx]
        
        # Get bottommost point (maximum y)
        bottom_idx = np.argmax(hair_pixels[0])
        bottom_y = hair_pixels[0][bottom_idx]
        bottom_x = hair_pixels[1][bottom_idx]
        
        # For more accurate horizontal position, use median x at top and bottom
        # This helps avoid outliers on the edges
        top_row_pixels = np.where(hair_mask[top_y, :] > 0)[0]
        if len(top_row_pixels) > 0:
            top_x = int(np.median(top_row_pixels))
        
        bottom_row_pixels = np.where(hair_mask[bottom_y, :] > 0)[0]
        if len(bottom_row_pixels) > 0:
            bottom_x = int(np.median(bottom_row_pixels))
        
        top_point = (int(top_x), int(top_y))
        bottom_point = (int(bottom_x), int(bottom_y))
        
        print(f"  ✓ Hair extremes found")
        print(f"    Top: {top_point}")
        print(f"    Bottom: {bottom_point}")
        
        return top_point, bottom_point
    
    def calculate_hair_length(
        self,
        top_point: Tuple[int, int],
        bottom_point: Tuple[int, int],
        pixels_per_cm: float,
        confidence: float = 1.0
    ) -> HairMeasurement:
        """
        Calculate hair length from top and bottom points.
        
        Args:
            top_point: (x, y) coordinates of hair top
            bottom_point: (x, y) coordinates of hair bottom
            pixels_per_cm: Calibration value from reference object
            confidence: Confidence score for the measurement
            
        Returns:
            HairMeasurement object with results
        """
        print("\n[3/3] Calculating hair length...")
        
        # Calculate pixel distance (vertical)
        length_pixels = bottom_point[1] - top_point[1]
        
        # Convert to centimeters
        length_cm = length_pixels / pixels_per_cm
        
        print(f"  ✓ Hair length calculated")
        print(f"    Pixels: {length_pixels:.1f} px")
        print(f"    Length: {length_cm:.2f} cm")
        
        return HairMeasurement(
            top_point=top_point,
            bottom_point=bottom_point,
            length_pixels=length_pixels,
            length_cm=length_cm,
            pixels_per_cm=pixels_per_cm,
            confidence=confidence
        )
    
    # ===== VISUALIZATION =====
    
    def visualize_measurement(
        self,
        image: np.ndarray,
        reference: Optional[ReferenceObject],
        measurement: Optional[HairMeasurement],
        hair_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create visualization of the measurement process.
        
        Args:
            image: Original image
            reference: Detected reference object
            measurement: Hair measurement result
            hair_mask: Optional hair segmentation mask for overlay
            
        Returns:
            Visualization image
        """
        print("\n[4/4] Creating visualization...")
        
        # Create copy of image
        vis = image.copy()
        
        # Draw hair mask overlay if provided
        if hair_mask is not None:
            # Create colored overlay (semi-transparent green)
            hair_overlay = np.zeros_like(vis)
            hair_overlay[hair_mask > 0] = [0, 255, 0]  # Green
            vis = cv2.addWeighted(vis, 0.7, hair_overlay, 0.3, 0)
        
        # Draw A4 paper contour
        if reference is not None:
            cv2.drawContours(vis, [reference.contour], -1, (0, 0, 255), 3)
            
            # Draw center point
            cv2.circle(vis, reference.center, 8, (0, 0, 255), -1)
            
            # Add calibration text
            calib_text = f"Calibration: {reference.pixels_per_cm:.2f} px/cm"
            cv2.putText(vis, calib_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw hair measurement
        if measurement is not None:
            # Draw top point (blue)
            cv2.circle(vis, measurement.top_point, 10, (255, 0, 0), -1)
            cv2.putText(vis, "Top", 
                       (measurement.top_point[0] + 15, measurement.top_point[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw bottom point (red)
            cv2.circle(vis, measurement.bottom_point, 10, (0, 0, 255), -1)
            cv2.putText(vis, "Bottom",
                       (measurement.bottom_point[0] + 15, measurement.bottom_point[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw measurement line
            cv2.line(vis, measurement.top_point, measurement.bottom_point, 
                    (255, 255, 0), 3)
            
            # Add length text at the bottom
            length_text = f"Hair Length: {measurement.length_cm:.2f} cm"
            text_size = cv2.getTextSize(length_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (vis.shape[1] - text_size[0]) // 2
            text_y = vis.shape[0] - 30
            
            # Draw background rectangle for text
            cv2.rectangle(vis, 
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(vis, length_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        print("  ✓ Visualization created")
        
        return vis
    
    # ===== VALIDATION =====
    
    def validate_measurement(
        self,
        measurement: HairMeasurement,
        min_length_cm: float = 1.0,
        max_length_cm: float = 200.0
    ) -> Tuple[bool, str]:
        """
        Validate measurement result.
        
        Args:
            measurement: Hair measurement to validate
            min_length_cm: Minimum reasonable hair length
            max_length_cm: Maximum reasonable hair length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if length is within reasonable range
        if measurement.length_cm < min_length_cm:
            return False, f"Hair length too short: {measurement.length_cm:.2f} cm (min: {min_length_cm} cm)"
        
        if measurement.length_cm > max_length_cm:
            return False, f"Hair length too long: {measurement.length_cm:.2f} cm (max: {max_length_cm} cm)"
        
        # Check if confidence is sufficient
        if measurement.confidence < self.min_confidence:
            return False, f"Confidence too low: {measurement.confidence:.2%} (min: {self.min_confidence:.2%})"
        
        return True, "Valid"


# ===== UTILITY FUNCTIONS =====

def create_default_config() -> Dict:
    """Create default configuration for HairLengthCV."""
    return {
        'detection_method': 'color',  # 'color', 'edges', or 'auto'
        'paper_color': 'yellow',      # Color of A4 paper
        'edge_threshold1': 50,
        'edge_threshold2': 150,
        'min_area_pixels': 10000,
        'aspect_ratio_tolerance': 0.15,
        'min_confidence': 0.7,
        'morph_kernel_size': 5
    }