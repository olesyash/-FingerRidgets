from abc_finger_region_recognizer import FingerRegionRecognizerTemplate
import cv2 as cv
from glob import glob
import os
import numpy as np
from utils.poincare import calculate_singularities
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import calculate_minutiaes
from tqdm import tqdm
from utils.skeletonize import skeletonize


class SecondImplementation(FingerRegionRecognizerTemplate):
    def __init__(self, block_size=16) -> None:
        super().__init__()
        self.block_size = block_size

    def normalize_image(self, input_img) -> str:
        print("Overwrite normalization")
        normalized_img = normalize(input_img.copy(), float(100), float(100))
        return normalized_img

    def segment_image(self, normalized_img, block_size=16) -> str:
        print("Overwrite segmentation")
        # color threshold
        # threshold_img = normalized_img
        # _, threshold_im = cv.threshold(normalized_img,127,255,cv.THRESH_OTSU)
        # cv.imshow('color_threshold', normalized_img); cv.waitKeyEx()

        # ROI and normalisation
        (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)
        return mask, normim, segmented_img

    # def lines_orientation(self, normalized_img, mask, segmented_img, block_size, smoth=False) -> str:
    #     # orientations
    #     angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=smoth)
    #     orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)
    #
    #     return angles, orientation_img

    # def calculate_ridge_frequencies(self, normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15) -> str:
    #     # find the overall frequency of ridges in Wavelet Domain
    #     freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
    #     return freq
    #
    # def apply_gabor_filter(self, normim, angles, freq) -> str:
    #     # create gabor filter and do the actual filtering
    #     gabor_img = gabor_filter(normim, angles, freq)
    #     return gabor_img
    #
    # def count_fingerprint_ridges(self, input_img):
    #     normalized_img = self.normalize_image(input_img)
    #     mask, normim, segmented_img = self.segment_image(normalized_img, self.block_size)
    #     angles, orientation_img = self.lines_orientation(normim, mask, segmented_img, 16, smoth=False)
    #
    #     # color threshold
    #     # threshold_img = normalized_img
    #     # _, threshold_im = cv.threshold(normalized_img,127,255,cv.THRESH_OTSU)
    #     # cv.imshow('color_threshold', normalized_img); cv.waitKeyEx()
    #
    #     # find the overall frequency of ridges in Wavelet Domain
    #     freq = ridge_freq(normim, mask, angles, self.block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
    #
    #     # create gabor filter and do the actual filtering
    #     gabor_img = gabor_filter(normim, angles, freq)
    #
    #     thin_image = self.skeletonize(gabor_img)
    #     minutiae_weights_image = self.calculate_minutiae_weights(thin_image)
    #
    #     block_size = 15  # 120/8
    #     best_region = self.get_best_region(thin_image, minutiae_weights_image, block_size, mask)
    #     result_image = self.draw_ridges_count_on_region(best_region, input_img, thin_image, block_size)
    #     return result_image
