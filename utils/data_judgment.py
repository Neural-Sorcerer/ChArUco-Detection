"""Data quality assessment module for camera calibration.

This module provides functionality for evaluating and filtering calibration data
to ensure diverse and high-quality samples for camera calibration.
"""
import os
import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from math import sqrt, pi, acos
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class CalibrationSample:
    """Represents a single calibration sample with its quality metrics."""
    corners: np.ndarray
    image_path: str
    timestamp: float
    x_position: float  # Normalized [0, 1]
    y_position: float  # Normalized [0, 1]
    size: float        # Normalized [0, 1]
    skew: float        # Normalized [0, 1]
    area: float        # Actual area in pixels
    angle: float       # Board angle in degrees
    is_accepted: bool = False


class DataQualityJudge:
    """Evaluates and filters calibration data for optimal diversity and quality."""
    
    def __init__(self, 
                 image_size: Tuple[int, int],
                 grid_size: Tuple[int, int] = (12, 8),
                 min_distance_threshold: float = 0.15,
                 min_size_ratio: float = 0.05,
                 max_size_ratio: float = 0.8,
                 max_skew: float = 0.7,
                 target_samples: int = 50):
        """Initialize the data quality judge.
        
        Args:
            image_size: Image dimensions (width, height)
            grid_size: Grid size for heatmap (cols, rows)
            min_distance_threshold: Minimum distance between samples in parameter space
            min_size_ratio: Minimum board size relative to image
            max_size_ratio: Maximum board size relative to image
            max_skew: Maximum allowed skew (0=no skew, 1=high skew)
            target_samples: Target number of diverse samples
        """
        self.image_size = image_size
        self.grid_size = grid_size
        self.min_distance_threshold = min_distance_threshold
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.max_skew = max_skew
        self.target_samples = target_samples
        
        # Storage for samples
        self.samples: List[CalibrationSample] = []
        self.accepted_samples: List[CalibrationSample] = []
        
        # Heatmap for position coverage
        self.position_heatmap = np.zeros(grid_size)
        
        # Progress tracking
        self.coverage_stats = {
            'x_coverage': 0.0,
            'y_coverage': 0.0,
            'size_coverage': 0.0,
            'angle_coverage': 0.0,
            'total_coverage': 0.0
        }
    
    def calculate_board_metrics(self, corners: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for a detected board.
        
        Args:
            corners: Detected corner points
            
        Returns:
            Dictionary with calculated metrics
        """
        # Get outside corners (assuming rectangular arrangement)
        corners_2d = corners.reshape(-1, 2)
        
        # Find bounding box
        min_x, min_y = np.min(corners_2d, axis=0)
        max_x, max_y = np.max(corners_2d, axis=0)
        
        # Calculate center position (normalized)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        x_pos = center_x / self.image_size[0]
        y_pos = center_y / self.image_size[1]
        
        # Calculate area and size
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        size = sqrt(area / (self.image_size[0] * self.image_size[1]))
        
        # Calculate skew (deviation from rectangle)
        skew = self._calculate_skew(corners_2d)
        
        # Calculate board angle
        angle = self._calculate_angle(corners_2d)
        
        return {
            'x_position': x_pos,
            'y_position': y_pos,
            'size': size,
            'skew': skew,
            'area': area,
            'angle': angle
        }
    
    def _calculate_skew(self, corners: np.ndarray) -> float:
        """Calculate skew metric for the board.
        
        Args:
            corners: Corner points as Nx2 array
            
        Returns:
            Skew value between 0 (no skew) and 1 (high skew)
        """
        if len(corners) < 4:
            return 1.0
        
        # Use the four corner points of bounding box
        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)
        
        # Find actual corners closest to bounding box corners
        corners_bbox = np.array([
            [min_x, min_y], [max_x, min_y],
            [max_x, max_y], [min_x, max_y]
        ])
        
        # Calculate angles at corners
        angles = []
        for i in range(4):
            p1 = corners_bbox[i]
            p2 = corners_bbox[(i + 1) % 4]
            p3 = corners_bbox[(i + 2) % 4]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = acos(abs(cos_angle))
            angles.append(abs(angle - pi/2))
        
        # Skew is the maximum deviation from 90 degrees
        max_deviation = max(angles)
        skew = min(1.0, 2.0 * max_deviation / (pi/2))
        
        return skew
    
    def _calculate_angle(self, corners: np.ndarray) -> float:
        """Calculate the rotation angle of the board.
        
        Args:
            corners: Corner points as Nx2 array
            
        Returns:
            Angle in degrees
        """
        if len(corners) < 2:
            return 0.0
        
        # Use first two corners to estimate orientation
        dx = corners[1, 0] - corners[0, 0]
        dy = corners[1, 1] - corners[0, 1]
        
        angle = np.arctan2(dy, dx) * 180 / pi
        return abs(angle) % 90  # Normalize to 0-90 degrees
    
    def evaluate_sample(self, corners: np.ndarray, image_path: str = "", timestamp: float = 0.0) -> CalibrationSample:
        """Evaluate a calibration sample and determine if it should be accepted.
        
        Args:
            corners: Detected corner points
            image_path: Path to the image file
            timestamp: Timestamp of the sample
            
        Returns:
            CalibrationSample object with evaluation results
        """
        metrics = self.calculate_board_metrics(corners)
        
        sample = CalibrationSample(
            corners=corners,
            image_path=image_path,
            timestamp=timestamp,
            **metrics
        )
        
        # Check basic quality criteria
        if not self._passes_basic_quality(sample):
            sample.is_accepted = False
            return sample
        
        # Check diversity criteria
        if not self._passes_diversity_check(sample):
            sample.is_accepted = False
            return sample
        
        # Sample passes all checks
        sample.is_accepted = True
        self._add_accepted_sample(sample)
        
        return sample
    
    def _passes_basic_quality(self, sample: CalibrationSample) -> bool:
        """Check if sample passes basic quality criteria.
        
        Args:
            sample: Sample to evaluate
            
        Returns:
            True if sample passes basic quality checks
        """
        # Size check
        if (sample.size < self.min_size_ratio) or (sample.size > self.max_size_ratio):
            logging.debug(f"Sample rejected: size {sample.size:.3f} outside range [{self.min_size_ratio}, {self.max_size_ratio}]")
            return False
        
        # Skew check
        if sample.skew > self.max_skew:
            logging.debug(f"Sample rejected: skew {sample.skew:.3f} > {self.max_skew}")
            return False
        
        # Position check (ensure board is not too close to edges)
        margin = 0.1
        if ((sample.x_position < margin) or (sample.x_position > (1 - margin)) or
            (sample.y_position < margin) or (sample.y_position > (1 - margin))):
            logging.debug(f"Sample rejected: too close to edges ({sample.x_position:.3f}, {sample.y_position:.3f})")
            return False
        
        return True
    
    def _passes_diversity_check(self, sample: CalibrationSample) -> bool:
        """Check if sample adds sufficient diversity to the dataset.
        
        Args:
            sample: Sample to evaluate
            
        Returns:
            True if sample adds sufficient diversity
        """
        if not self.accepted_samples:
            return True
        
        # Calculate parameter vector for this sample
        params = np.array([sample.x_position, sample.y_position, sample.size, sample.skew])
        
        # Check distance to all existing samples
        for existing_sample in self.accepted_samples:
            existing_params = np.array([
                existing_sample.x_position,
                existing_sample.y_position,
                existing_sample.size,
                existing_sample.skew
            ])
            
            # Calculate Manhattan distance in parameter space
            distance = np.sum(np.abs(params - existing_params))
            
            if distance < self.min_distance_threshold:
                logging.debug(f"Sample rejected: too similar to existing sample (distance: {distance:.3f})")
                return False
        
        return True
    
    def _add_accepted_sample(self, sample: CalibrationSample) -> None:
        """Add an accepted sample and update statistics.
        
        Args:
            sample: Accepted sample to add
        """
        self.accepted_samples.append(sample)
        
        # Update position heatmap
        grid_x = min(int(sample.x_position * self.grid_size[0]), self.grid_size[0] - 1)
        grid_y = min(int(sample.y_position * self.grid_size[1]), self.grid_size[1] - 1)
        self.position_heatmap[grid_x, grid_y] += 1
        
        # Update coverage statistics
        self._update_coverage_stats()
        
        logging.info(f"Sample accepted: pos=({sample.x_position:.3f}, {sample.y_position:.3f}), "
                     f"size={sample.size:.3f}, skew={sample.skew:.3f}, angle={sample.angle:.1f}°")
    
    def _update_coverage_stats(self) -> None:
        """Update coverage statistics based on accepted samples."""
        if not self.accepted_samples:
            return
        
        # Extract parameter arrays
        x_positions = [s.x_position for s in self.accepted_samples]
        y_positions = [s.y_position for s in self.accepted_samples]
        sizes = [s.size for s in self.accepted_samples]
        angles = [s.angle for s in self.accepted_samples]
        
        # Calculate coverage as range of values
        self.coverage_stats['x_coverage'] = max(x_positions) - min(x_positions)
        self.coverage_stats['y_coverage'] = max(y_positions) - min(y_positions)
        self.coverage_stats['size_coverage'] = max(sizes) - min(sizes)
        self.coverage_stats['angle_coverage'] = max(angles) - min(angles)
        
        # Total coverage as average
        self.coverage_stats['total_coverage'] = np.mean([
            self.coverage_stats['x_coverage'],
            self.coverage_stats['y_coverage'],
            self.coverage_stats['size_coverage'],
            self.coverage_stats['angle_coverage'] / 90.0  # Normalize angle to [0,1]
        ])
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information.
        
        Returns:
            Dictionary with progress information
        """
        return {
            'accepted_samples': len(self.accepted_samples),
            'target_samples': self.target_samples,
            'progress_ratio': len(self.accepted_samples) / self.target_samples,
            'coverage_stats': self.coverage_stats.copy(),
            'is_sufficient': len(self.accepted_samples) >= self.target_samples
        }
    
    def generate_heatmap(self, save_path: Optional[str] = None) -> np.ndarray:
        """Generate a heatmap showing board position coverage.
        
        Args:
            save_path: Optional path to save the heatmap image
            
        Returns:
            Heatmap as numpy array
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(self.position_heatmap, cmap='hot', interpolation='nearest')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Number of samples')
        
        # Set labels and title
        ax.set_xlabel('X Position (normalized)')
        ax.set_ylabel('Y Position (normalized)')
        ax.set_title(f'Calibration Board Position Coverage\n({len(self.accepted_samples)} samples)')
        
        # Add grid
        ax.set_xticks(range(self.grid_size[0]))
        ax.set_yticks(range(self.grid_size[1]))
        ax.grid(True, alpha=0.3)
        
        # Add sample count annotations
        for i in range(self.grid_size[1]):  # rows (y-axis)
            for j in range(self.grid_size[0]):  # cols (x-axis)
                count = int(self.position_heatmap[j, i])
                if count > 0:
                    ax.text(i, j, str(count), ha='center', va='center',
                           color='white' if count > np.max(self.position_heatmap) / 2 else 'black')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logging.info(f"Heatmap saved to {save_path}")
        
        # Convert to numpy array
        fig.canvas.draw()
        try:
            # Try new method first (matplotlib >= 3.0)
            buf = fig.canvas.buffer_rgba()
            heatmap_array = np.asarray(buf)
            # Convert RGBA to RGB
            heatmap_array = heatmap_array[:, :, :3]
        except AttributeError:
            # Fallback to older method
            try:
                heatmap_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                heatmap_array = heatmap_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # Last resort - use buffer_rgba and convert
                heatmap_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                heatmap_array = heatmap_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                heatmap_array = heatmap_array[:, :, :3]  # Remove alpha channel
        
        plt.close(fig)
        return heatmap_array

    def render_progress_overlay(self, image: np.ndarray) -> np.ndarray:
        """Render progress information overlay on image.

        Args:
            image: Input image to overlay progress on

        Returns:
            Image with progress overlay
        """
        # Create extended frame for progress info
        overlay_width = 350
        frame = np.ones((image.shape[0], image.shape[1] + overlay_width, 3), dtype=np.uint8) * 255
        frame[:image.shape[0], :image.shape[1], :] = image.copy()

        # Get progress info
        progress = self.get_progress_info()
        coverage = progress['coverage_stats']

        # Draw progress bars and text
        y_offset = 30
        line_height = 70

        # Sample count
        cv2.putText(frame, f'Samples: {progress["accepted_samples"]}/{progress["target_samples"]}',
                   (image.shape[1] + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Coverage metrics
        metrics = [
            ('X Coverage', coverage['x_coverage']),
            ('Y Coverage', coverage['y_coverage']),
            ('Size Coverage', coverage['size_coverage']),
            ('Angle Coverage', coverage['angle_coverage'] / 90.0)
        ]

        for i, (name, value) in enumerate(metrics):
            y_pos = y_offset + (i + 1) * line_height

            # Text
            cv2.putText(frame, f'{name}: {value:.1%}',
                       (image.shape[1] + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Progress bar
            bar_y = y_pos + 10
            bar_width = 200
            bar_height = 15

            # Background bar
            cv2.rectangle(frame,
                         (image.shape[1] + 10, bar_y),
                         (image.shape[1] + 10 + bar_width, bar_y + bar_height),
                         (200, 200, 200), -1)

            # Progress bar
            progress_width = int(bar_width * min(value, 1.0))
            color = (0, 255, 0) if value >= 0.8 else (0, 255, 255) if value >= 0.5 else (0, 0, 255)
            cv2.rectangle(frame,
                         (image.shape[1] + 10, bar_y),
                         (image.shape[1] + 10 + progress_width, bar_y + bar_height),
                         color, -1)

        # Status message
        status_y = y_offset + len(metrics) * line_height + 50
        if progress['is_sufficient']:
            status_text = "Ready for calibration!"
            status_color = (0, 255, 0)
        else:
            status_text = "Collecting diverse samples..."
            status_color = (0, 0, 255)

        cv2.putText(frame, status_text,
                   (image.shape[1] + 10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return frame

    def filter_existing_dataset(self, samples: List[CalibrationSample]) -> List[CalibrationSample]:
        """Filter an existing dataset to remove redundant samples.

        Args:
            samples: List of calibration samples to filter

        Returns:
            Filtered list of diverse samples
        """
        if not samples:
            return []

        # Reset accepted samples
        self.accepted_samples = []
        self.position_heatmap = np.zeros(self.grid_size)

        # Sort samples by quality score (size * (1 - skew))
        samples_sorted = sorted(samples,
                               key=lambda s: s.size * (1 - s.skew),
                               reverse=True)

        filtered_samples = []

        for sample in samples_sorted:
            # Re-evaluate sample for diversity
            if self._passes_basic_quality(sample) and self._passes_diversity_check(sample):
                self._add_accepted_sample(sample)
                filtered_samples.append(sample)

                # Stop if we have enough samples
                if len(filtered_samples) >= self.target_samples:
                    break

        logging.info(f"Filtered dataset: {len(samples)} -> {len(filtered_samples)} samples")
        return filtered_samples

    def export_summary(self, output_path: str) -> None:
        """Export a summary of the calibration data quality.

        Args:
            output_path: Path to save the summary file
        """
        progress = self.get_progress_info()

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj

        summary = {
            'total_samples': len(self.accepted_samples),
            'target_samples': self.target_samples,
            'coverage_stats': convert_numpy_types(progress['coverage_stats']),
            'quality_metrics': {
                'avg_size': float(np.mean([s.size for s in self.accepted_samples])) if self.accepted_samples else 0.0,
                'avg_skew': float(np.mean([s.skew for s in self.accepted_samples])) if self.accepted_samples else 0.0,
                'size_range': (
                    float(np.min([s.size for s in self.accepted_samples])) if self.accepted_samples else 0.0,
                    float(np.max([s.size for s in self.accepted_samples])) if self.accepted_samples else 0.0
                ),
                'angle_range': (
                    float(np.min([s.angle for s in self.accepted_samples])) if self.accepted_samples else 0.0,
                    float(np.max([s.angle for s in self.accepted_samples])) if self.accepted_samples else 0.0
                )
            },
            'recommendations': self._generate_recommendations()
        }

        # Save as JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logging.info(f"Summary exported to {output_path}")

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving data collection.

        Returns:
            List of recommendation strings
        """
        recommendations = []
        coverage = self.coverage_stats

        if coverage['x_coverage'] < 0.6:
            recommendations.append("Move the board more across the horizontal axis")

        if coverage['y_coverage'] < 0.6:
            recommendations.append("Move the board more across the vertical axis")

        if coverage['size_coverage'] < 0.4:
            recommendations.append("Vary the distance to the board more (closer and farther)")

        if coverage['angle_coverage'] < 30:
            recommendations.append("Tilt the board at different angles")

        if len(self.accepted_samples) < self.target_samples * 0.5:
            recommendations.append("Collect more diverse samples")

        # Check heatmap for cold spots
        if np.min(self.position_heatmap) == 0:
            recommendations.append("Cover all regions of the image (check heatmap for gaps)")

        if not recommendations:
            recommendations.append("Data collection looks good! Ready for calibration.")

        return recommendations
