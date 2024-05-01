from abc_finger_region_recognizer import FingerRegionRecognizerTemplate
import cv2
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
from utils.skeletonize import skeletonize


class SecondImplementation(FingerRegionRecognizerTemplate):
    def __init__(self, image, block_size=16) -> None:
        super().__init__(image)
        self.image = image
        self.block_size = block_size

    def first_step(self) -> cv2.typing.MatLike:
        self.normalized_img = normalize(self.image.copy(), float(100), float(100))
        self.segmented_image, self.normim, self.mask = self.__segment_image(
            self.normalized_img, self.block_size
        )
        return self.segmented_image

    def __segment_image(
        self, normalized_img, block_size
    ) -> tuple[cv2.typing.MatLike, np.ndarray]:
        # ROI and normalisation
        (segmented_img, normim, mask) = create_segmented_and_variance_images(
            normalized_img, block_size, 0.2
        )
        return segmented_img, normim, mask

    def __lines_orientation(
        self, normalized_img, mask, segmented_img, block_size, smoth=False
    ) -> str:
        # orientations
        angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=smoth)
        orientation_img = orientation.visualize_angles(
            segmented_img, mask, angles, W=block_size
        )

        return angles, orientation_img

    def second_step(self) -> cv2.typing.MatLike:
        self.angles, self.orientation = self.__lines_orientation(
            self.normim,
            self.mask,
            self.segmented_image,
            self.block_size,
            smoth=False,
        )
        return self.orientation

    def __calculate_ridge_frequencies(
        self,
        normim,
        mask,
        angles,
        block_size,
        kernel_size=5,
        minWaveLength=5,
        maxWaveLength=15,
    ) -> str:
        # find the overall frequency of ridges in Wavelet Domain
        return ridge_freq(
            normim,
            mask,
            angles,
            block_size,
            kernel_size=5,
            minWaveLength=5,
            maxWaveLength=15,
        )

    def third_step(self) -> cv2.typing.MatLike:
        self.frequency = self.__calculate_ridge_frequencies(
            self.normim, self.mask, self.angles, self.block_size
        )
        return self.frequency

    def fourth_step(self) -> cv2.typing.MatLike:
        self.gabor_img = gabor_filter(self.normim, self.angles, self.frequency)
        return self.gabor_img
