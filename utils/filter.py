
# === Standard Libraries ===
import os
from typing import *
from enum import Enum
from xml.dom import minidom
import xml.etree.ElementTree as ET
from math import sqrt, ceil, pi, acos

# === Third-Party Libraries ===
import cv2
import numpy as np


# Camera calibration flags
PINHOLE_CALIBRATION_FLAGS = 0
FISHEYE_CALIBRATION_FLAGS = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW

# Corner flags
RGB_CORNER_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
THERMAL_CORNER_FLAG = cv2.CALIB_CB_ADAPTIVE_THRESH

# Criterias for cv2.cornerSubPix()
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
DEFAULT_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# CLAHE for thermal sensor
CLAHE = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8, 8))

PARAM_RANGES = [0.7, 0.7, 0.4, 0.5]


def _pdist(p1, p2):
    """
    L-2 distance of two points
    
    Args:
        p1 = (x1, y1)
        p2 = (x2, y2)    
    """
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


def _calculate_skew(corners):
    """
    Get skew for given checkerboard detection.
    Scaled to [0, 1], where 0 = no skew, 1 = high skew
    Skew is proportional to the divergence of three outside corners from 90 degrees.

    Note: TODO
        Using three nearby interior corners might be more robust,
        outside corners occasionally get mis-detected
    """
    up_left, up_right, down_right, _ = corners

    def angle(a, b, c):
        """ Return angle between lines ab, bc """
        ab = a - b
        cb = c - b
        return acos(np.dot(ab,cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)))

    skew = min(1.0, 2. * abs((pi / 2.) - angle(up_left, up_right, down_right)))
    return skew


def _calculate_area(corners):
    """ Get 2D image area of the detected checkerboard.
    The projected checkerboard is assumed to be a convex quadrilateral, and the area computed as
    |p X q|/2; see http://mathworld.wolfram.com/Quadrilateral.html.
    """
    (up_left, up_right, down_right, down_left) = corners
    a = up_right - up_left
    b = down_right - up_right
    c = down_left - down_right
    p = b + c
    q = a + b
    return abs(p[0]*q[1] - p[1]*q[0]) / 2.0


def lmin(seq1, seq2):
    """ Pairwise minimum of two sequences """
    return [min(a, b) for (a, b) in zip(seq1, seq2)]


def lmax(seq1, seq2):
    """ Pairwise maximum of two sequences """
    return [max(a, b) for (a, b) in zip(seq1, seq2)]


def renderProgress(image, progress, isGoodEnough):
    frame = np.ones((image.shape[0], image.shape[1] + 310, 3), dtype=np.uint8) * 255
    frame[:image.shape[0], :image.shape[1], :] = image.copy()
    
    cv2.putText(frame, 'X: {:.3f}%'.format(progress[0] * 100), (image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(frame, 'Y: {:.3f}%'.format(progress[1] * 100), (image.shape[1] + 10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(frame, 'Size: {:.3f}%'.format(progress[2] * 100), (image.shape[1] + 10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(frame, 'Skew: {:.3f}%'.format(progress[3] * 100), (image.shape[1] + 10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    
    if int(progress[0]) == 1:
        cv2.line(frame, (image.shape[1] + 10, 40), (image.shape[1] + 10 + 200, 40), (0, 255, 0), 4)
    else:
         cv2.line(frame, (image.shape[1] + 10, 40), (image.shape[1] + 10 + int(progress[0] * 200), 40), (0, 255, 255), 4)
    
    if int(progress[1]) == 1:
        cv2.line(frame, (image.shape[1] + 10, 110), (image.shape[1] + 10 + 200, 110), (0, 255, 0), 4)
    else:
         cv2.line(frame, (image.shape[1] + 10, 110), (image.shape[1] + 10 + int(progress[1] * 200), 110), (0, 255, 255), 4)

    if int(progress[2]) == 1:
        cv2.line(frame, (image.shape[1] + 10, 180), (image.shape[1] + 10 + 200, 180), (0, 255, 0), 4)
    else:
         cv2.line(frame, (image.shape[1] + 10, 180), (image.shape[1] + 10 + int(progress[2] * 200), 180), (0, 255, 255), 4)

    if int(progress[3]) == 1:
        cv2.line(frame, (image.shape[1] + 10, 250), (image.shape[1] + 10 + 200, 250), (0, 255, 0), 4)
    else:
         cv2.line(frame, (image.shape[1] + 10, 250), (image.shape[1] + 10 + int(progress[3] * 200), 250), (0, 255, 255), 4)
    
    if isGoodEnough:
        text = "Press 'Q' to calibrate!"
    else:
        text = "Accumulating images..."
    
    cv2.putText(frame, text, (image.shape[1] + 10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    return frame


class CAMERA_MODEL(Enum):
    """ Supported camera model

    Args:
        PINHOLE = 0
        FISHEYE = 1
    """    
    PINHOLE = 0
    FISHEYE = 1


class IMAGE_SENSOR(Enum):
    """ Supported sensor

    Args:
        RGB = 0 
        THERMAL = 1
    """    
    RGB = 0
    THERMAL = 1


class CornerDetector:
    """Corner Detector class:
    
    Properties:
        img_shape: image shape
        nRows: checkerboard's number of rows
        nCols: checkerboard's number of columns
        nBorders: number of border pixels
        is_low_res: working on low resolution (True) or high resolution (False) stream 
    """    
    def __init__(self, checkerboard, is_low_res, border=8):
        """ Instance initialization

        Args:
            checkerboard (tuple): checkerboard shape
            is_low_res (bool): working on low resolution (True) or high resolution (False) stream 
            border (int): number of border pixels (default is 8)
        """        
        self.img_shape = None
        self.nBorders = border
        self.is_low_res = is_low_res
        
        # Make sure nCols > nRows to agree with OpenCV CB detector output
        self.nCols = max(checkerboard[0], checkerboard[1])
        self.nRows = min(checkerboard[0], checkerboard[1])
            
    def getOutsideCorners(self, corners):
        """ Return the four corners of the board.

        Args:
            corners (list): input corners

        Raises:
            Exception: Invalid number of corners

        Returns:
            (up_left, up_right, down_right, down_left)
        """
        nColsRows = self.nCols * self.nRows
        nCorners = corners.shape[1] * corners.shape[0]
        
        if nCorners != nColsRows:
            raise Exception(f"Invalid number of corners! {nCorners} corners. X: {self.nCols}, Y: {self.nRows}")

        up_left = corners[0, 0]
        up_right = corners[self.nCols-1, 0]
        down_right = corners[-1, 0]
        down_left = corners[-self.nCols, 0]
        return (up_left, up_right, down_right, down_left)
    
    def getCorners(self, img, isThermal):
        """ Get corners from the image.

        Args:
            img: input image to get corners
            isThermal (bool): image is thermal image (True) or not (False)
        Returns:
            retval: result (True if corners are extracted successfully, False otherwise)
            corners: corner list
        """        
        if self.img_shape == None:
            self.img_shape = img.shape[:2][::-1]
        else:
            assert self.img_shape == img.shape[:2][::-1], "All images must share the same size."
            
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        if isThermal:
            gray = np.array(255 - gray, dtype=np.uint8)
            gray = CLAHE.apply(gray)
            flags = THERMAL_CORNER_FLAG
        else:
            flags = RGB_CORNER_FLAGS
            
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (self.nCols, self.nRows), flags=flags)
        
        if not ret:
            ret, corners = cv2.findChessboardCornersSB(gray, (self.nCols, self.nRows), flags=cv2.CALIB_CB_EXHAUSTIVE)
            if not ret:
                return ret, corners
        
        # Reject detection if any corner is within BORDER pixels of the image edge.
        if not self.is_low_res:
            valid_corners = all(
                self.nBorders < x < (self.img_shape[0] - self.nBorders) and
                self.nBorders < y < (self.img_shape[1] - self.nBorders)
                for x, y in corners[:, 0, :]
            )
            if not valid_corners:
                ret = False

        # Ensure chessboard corners are consistently ordered from top-left to bottom-right
        if self.nCols != self.nRows:
            # For non-square boards, flip if the first corner is below the last
            if corners[0, 0, 1] > corners[-1, 0, 1]:
                corners = np.copy(np.flipud(corners))
        else:
            # Determine the corner order direction
            direction_corners = (corners[-1] - corners[0]) >= np.array([[0.0, 0.0]])
            
            if not np.all(direction_corners):
                if not np.any(direction_corners):
                    corners = np.copy(np.flipud(corners))
                else:
                    # Rotate square board corners if needed
                    rotation_angle = 1 if direction_corners[0][0] else 3
                    corners = np.rot90(corners.reshape(self.nRows, self.nCols, 2), rotation_angle).reshape(-1, 1, 2)
        
        if ret:
            """ TODO
            The choice of radius depends on:
                Image resolution (higher resolution → larger radius).
                Chessboard square size (smaller squares → smaller radius).
                Noise level (noisy images need a slightly larger radius).
                
                Note: Use a radius of half the minimum distance between corners. This should be large enough
                to snap to the correct corner, but not so large as to include a wrong corner in the search window.
            """
            min_distance = float("inf")
            for row in range(self.nRows):
                for col in range(self.nCols - 1):
                    index = row * self.nRows + col
                    min_distance = min(min_distance, _pdist(corners[index, 0], corners[index + 1, 0]))
            
            for row in range(self.nRows - 1):
                for col in range(self.nCols):
                    index = row * self.nRows + col
                    min_distance = min(min_distance, _pdist(corners[index, 0], corners[index + self.nCols, 0]))
            radius = int(ceil(min_distance * 0.5))
        
            # Limit the radius [3, 11]
            radius = max(3, min(radius, 11))
            
            # Refine detected corners for better accuracy
            corners = cv2.cornerSubPix(gray, corners, (radius, radius), (-1, -1), SUBPIX_CRITERIA)
        return ret, corners


class Calibrator:
    """ Calibrator class. """    
    def __init__(self, param_thres=0.2, quantity_thres=40, square_size=0.035, is_fisheye=False, is_thermal=False):
        """Instance initialization
        Args:
            model (int): camera model (0: pinhole, 1: fisheye)
            param_thres (float): param threshold for evaluating good detected corners
            quantity_thres (int): minimum number of good images for calibration
        """
        self.param_thres = param_thres
        self.quantity_thres = quantity_thres
        self.square_size = square_size
        self.is_fisheye = is_fisheye
        self.is_thermal = is_thermal
        
        if is_fisheye:
            self.camera_model = CAMERA_MODEL.FISHEYE
            self.calibration_flags = FISHEYE_CALIBRATION_FLAGS
        else:
            self.camera_model = CAMERA_MODEL.PINHOLE
            self.calibration_flags = PINHOLE_CALIBRATION_FLAGS
            
        if is_thermal:
            self.image_sensor = IMAGE_SENSOR.THERMAL
        else:
            self.image_sensor = IMAGE_SENSOR.RGB
        
        self.dim = None
        self.progress = [0., 0., 0., 0.]

        # self.db is list of parameters samples for use in calibration. Parameters has form
        # (X, Y, size, skew) all normalized to [0, 1], to keep track of what sort of samples we've taken
        # and ensure enough variety.
        self.db = list()
        self.good_corners = list()
        self.last_frame_corners = None
        self.good_enough = False
        self.param_ranges = PARAM_RANGES
        #self._param_names = ["X", "Y", "Size", "Skew"]
        
    def setCheckerboard(self, checkerboard, is_low_res=True, border=8):
        """ Set checkerboard shape for calibration.

        Args:
            checkerboard: checkerboard shape
            is_low_res: stream is low resolution (True) or not (False)
            border (int): number of border pixels (default is 8)
        """        
        self.detector = CornerDetector(checkerboard, is_low_res, border)

    def getParams(self, corners):
        """
        Return list of parameters [X, Y, size, skew] describing the checkerboard view:
        
        Args:
            corners: detected corners from this view
        """
        (height, width) = self.dim
        Xs = corners[:, :, 0]
        Ys = corners[:, :, 1]
        outside_corners = self.detector.getOutsideCorners(corners)
        area = _calculate_area(outside_corners)
        skew = _calculate_skew(outside_corners)
        border = sqrt(area)
        # For X and Y, we "shrink" the image all around by approx. half the board size.
        # Otherwise large boards are penalized because you can't get much X/Y variation.
        p_x = min(1.0, max(0.0, (np.mean(Xs) - border / 2) / (width  - border)))
        p_y = min(1.0, max(0.0, (np.mean(Ys) - border / 2) / (height - border)))
        p_size = sqrt(area / (width * height))
        params = [p_x, p_y, p_size, skew]
        return params
        
    def isGoodSample(self, params):
        """ Return true if the checkerboard detection described by params should be added to the database. """
        if not self.db:
            return True

        def param_distance(p1, p2):
            """ Distance between 2 params (Manhattan distance).

            Args:
                p1: 1st param
                p2: 2nd param

            Returns:
                distance
            """            
            return sum([abs(a - b) for (a, b) in zip(p1, p2)])

        db_params = [sample for sample in self.db]
        d = min([param_distance(params, p) for p in db_params])
        
        # print "d = %.3f" % d #DEBUG
        # TODO What's a good threshold here? Should it be configurable?
        if d <= self.param_thres:
            return False

        # All tests passed, image should be good for calibration
        return True
    
    def detectCorners(self, img, accumulate=True):
        """ Detect corners from image.

        Args:
            img: input image
            accumulate: accumulate the detected corners for calibration or not (default is True)
        
        Return:
            retval:
                -1: no corner detected.
                0: "good" corner detected.
                1: "bad" corner (not good for accum) detected.
            corners: detected corners.
            img: original image in case no corner detected, otherwise is the image with detected corner.
            progress: calibration progress (only return when accum=True).
        """ 
        retval = -1
        success, corners = self.detector.getCorners(img, self.is_thermal)
        
        if success:
            if self.dim is None:
                self.dim = self.detector.img_shape
            
            # Add sample to database only if it's sufficiently different from any previous sample.
            if accumulate:
                params = self.getParams(corners)
                if self.isGoodSample(params):
                    retval = 0
                    self.db.append(params)
                    self.good_corners.append(corners)
                    self.last_frame_corners = corners

                    if len(self.db) == 1:
                        self.min_params = params
                        self.max_params = params
                    else:
                        self.min_params = lmin(self.min_params, params)
                        self.max_params = lmax(self.max_params, params)
                    
                    # Don't reward small size or skew
                    min_params = [self.min_params[0], self.min_params[1], 0., 0.]

                    # Assess each parameter to see how much variation has been achieved.
                    self.progress = [min((hi - lo) / r, 1.0) for (lo, hi, r) in zip(min_params, self.max_params, self.param_ranges)]
                    
                    # If we have a large number of samples, permit calibration even if not all parameters are optimal.
                    self.good_enough = (len(self.db) >= self.quantity_thres) or all([p == 1.0 for p in self.progress])
                else:
                    retval = 1
                    success = False
            else:
                retval = 0

            # Draw and display corners
            cv2.drawChessboardCorners(img, (self.detector.nCols, self.detector.nRows), corners, success)
        
        if accumulate:
            return retval, corners, img, self.progress
        
        return retval, corners, img


