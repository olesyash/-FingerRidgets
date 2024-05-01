import numpy as np
import cv2
from skimage.morphology import skeletonize as ski_skeletonize
import os
from scipy import ndimage
from scipy.ndimage import rotate, grey_dilation
import argparse
from abc_finger_region_recognizer import FingerRegionRecognizerTemplate


class ConcreteFingerRegionRecognizer(FingerRegionRecognizerTemplate):
    def __init__(self, image: cv2.typing.MatLike, block_size=16) -> None:
        super().__init__(image)
        self.image = image
        self.block_size = block_size

    def first_step(self) -> cv2.typing.MatLike:
        cropped_img = self.image[:-32, :]
        self.img_normalized = (cropped_img - np.min(cropped_img)) / (
            np.max(cropped_img) - np.min(cropped_img)
        )
        self.normalized_img, self.mask = self.__segment_image(
            self.img_normalized, self.block_size
        )
        return self.normalized_img

    def __segment_image(self, img_normalized, block_size=16, std_threshold=0.05):
        # Iterate over blocks and checks the std_dev
        rows, cols = img_normalized.shape
        mask = np.zeros((rows, cols), dtype=bool)
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                block = img_normalized[i : i + block_size, j : j + block_size]
                std_dev = np.std(block)
                # if higher than threshold, means it's a ROI
                mask[i : i + block_size, j : j + block_size] = std_dev > std_threshold

        # normalize ROIs
        masked_image = img_normalized[mask]
        # normalize whole image
        img_norm_masked = (img_normalized - np.mean(masked_image)) / np.std(
            masked_image
        )
        return img_norm_masked, mask

    def __lines_orientation(self, img, grad_rate=1, block_rate=7, orient_smooth_rate=7):
        gradient_kernel = cv2.getGaussianKernel(np.int_(6 * grad_rate + 1), grad_rate)
        gradient_matrix = gradient_kernel * gradient_kernel.T
        gy, gx = np.gradient(gradient_matrix)
        gradient_x = ndimage.convolve(img, gx, mode="constant", cval=0.0)
        gradient_y = ndimage.convolve(img, gy, mode="constant", cval=0.0)
        Gxx, Gyy, Gxy = gradient_x**2, gradient_y**2, gradient_x * gradient_y
        block_kernel = cv2.getGaussianKernel(6 * block_rate, block_rate)
        block_matrix = block_kernel * block_kernel.T
        Gxx = ndimage.convolve(Gxx, block_matrix, mode="constant", cval=0.0)
        Gyy = ndimage.convolve(Gyy, block_matrix, mode="constant", cval=0.0)
        Gxy = 2 * ndimage.convolve(Gxy, block_matrix, mode="constant", cval=0.0)
        determinant = np.sqrt((Gxx - Gyy) ** 2 + 4 * Gxy**2)
        determinant = np.where(determinant == 0, np.finfo(float).eps, determinant)
        determinant += np.finfo(float).eps
        sin_2theta = Gxy / determinant
        cos_2theta = (Gxx - Gyy) / determinant
        if orient_smooth_rate:
            smooth_kernel = cv2.getGaussianKernel(
                np.int_(6 * orient_smooth_rate + 1), orient_smooth_rate
            )
            smooth_matrix = smooth_kernel * smooth_kernel.T
            cos_2theta = ndimage.convolve(
                cos_2theta, smooth_matrix, mode="constant", cval=0.0
            )
            sin_2theta = ndimage.convolve(
                sin_2theta, smooth_matrix, mode="constant", cval=0.0
            )
        orientation = np.pi / 2 + np.arctan2(sin_2theta, cos_2theta) / 2
        return orientation

    def second_step(self) -> cv2.typing.MatLike:
        self.orientation = self.__lines_orientation(self.normalized_img)
        return self.orientation

    def third_step(self) -> cv2.typing.MatLike:
        self.frequency = self.__calculate_ridge_frequencies(
            self.normalized_img, self.orientation, self.mask
        )
        return self.frequency

    def __calculate_ridge_frequencies(
        self,
        image,
        orientation,
        mask,
        block_size=38,
        window_size=5,
        min_wavelength=5,
        max_wavelength=15,
    ):

        frequencies = np.zeros_like(image)

        for row in range(0, frequencies.shape[0] - block_size, block_size):
            for col in range(0, frequencies.shape[1] - block_size, block_size):
                block_image = image[row : row + block_size, col : col + block_size]
                block_orientation = orientation[
                    row : row + block_size, col : col + block_size
                ]
                frequencies[row : row + block_size, col : col + block_size] = (
                    self.calculate_block_frequencies(
                        block_image,
                        block_orientation,
                        window_size,
                        min_wavelength,
                        max_wavelength,
                    )
                )
        masked_frequencies = frequencies * mask
        non_zero_elements = masked_frequencies[mask > 0]
        mean_frequency = np.mean(non_zero_elements)

        return mean_frequency * mask

    def calculate_block_frequencies(
        self,
        block_image,
        block_orientation,
        window_size,
        min_wavelength,
        max_wavelength,
    ):

        cos_orient = np.mean(np.cos(2 * block_orientation))
        sin_orient = np.mean(np.sin(2 * block_orientation))
        orientation = np.arctan2(sin_orient, cos_orient) / 2

        rotated_image = rotate(
            block_image,
            orientation / np.pi * 180 + 90,
            axes=(1, 0),
            reshape=False,
            order=3,
            mode="nearest",
        )
        crop_size = np.sqrt(block_image.shape[0] * block_image.shape[1] / 2).astype(
            np.int_
        )
        offset = (block_image.shape[0] - crop_size) // 2
        cropped_image = rotated_image[
            offset : offset + crop_size, offset : offset + crop_size
        ]

        projection = np.sum(cropped_image, axis=0)
        dilation = grey_dilation(
            projection, window_size, structure=np.ones(window_size)
        )
        noise = np.abs(dilation - projection)
        peak_thresh = 2
        max_pts = (noise < peak_thresh) & (projection > np.mean(projection))
        max_ind = np.where(max_pts)
        num_peaks = len(max_ind[0])
        if num_peaks < 2:
            return np.zeros(block_image.shape)
        else:
            wavelength = (max_ind[0][-1] - max_ind[0][0]) / (num_peaks - 1)
            if min_wavelength <= wavelength <= max_wavelength:
                return 1 / np.double(wavelength) * np.ones(block_image.shape)
            else:
                return np.zeros(block_image.shape)

    def fourth_step(self) -> cv2.typing.MatLike:
        self.gabor_img = self.__apply_gabor_filter(
            self.normalized_img, self.frequency, self.orientation
        )
        return self.gabor_img

    def __apply_gabor_filter(
        self,
        image,
        frequency,
        orientation,
        threshold=-2,
        kx=0.65,
        ky=0.65,
        angle_increment=3,
    ):

        image = image.astype(np.float64)
        rows, cols = image.shape
        filtered_image = np.zeros((rows, cols))

        non_zero_freqs = frequency.ravel()[frequency.ravel() > 0]
        rounded_freqs = np.round(non_zero_freqs * 100) / 100
        unique_freqs = np.unique(rounded_freqs)

        sigma_x = 1 / unique_freqs[0] * kx
        sigma_y = 1 / unique_freqs[0] * ky
        filter_size = np.int_(np.round(3 * np.max([sigma_x, sigma_y])))

        x, y = np.meshgrid(
            np.arange(-filter_size, filter_size + 1),
            np.arange(-filter_size, filter_size + 1),
        )
        exponent = ((x / sigma_x) ** 2 + (y / sigma_y) ** 2) / 2
        reference_filter = np.exp(-exponent) * np.cos(2 * np.pi * unique_freqs[0] * x)
        angle_range = int(180 / angle_increment)
        gabor_filters = np.array(
            [
                rotate(reference_filter, -(o * angle_increment + 90), reshape=False)
                for o in range(angle_range)
            ]
        )

        max_size = filter_size
        valid_rows, valid_cols = np.where(frequency > 0)
        valid_indices = np.where(
            (valid_rows > max_size)
            & (valid_rows < rows - max_size)
            & (valid_cols > max_size)
            & (valid_cols < cols - max_size)
        )[0]
        max_orient_index = int(np.round(180 / angle_increment))
        orient_index = np.round(orientation / np.pi * 180 / angle_increment).astype(
            np.int32
        )
        orient_index[orient_index < 1] += max_orient_index
        orient_index[orient_index > max_orient_index] -= max_orient_index

        for k in valid_indices:
            r = valid_rows[k]
            c = valid_cols[k]
            img_block = image[
                r - filter_size : r + filter_size + 1,
                c - filter_size : c + filter_size + 1,
            ]
            filtered_image[r, c] = np.sum(
                img_block * gabor_filters[orient_index[r, c] - 1]
            )

        binary_image = (filtered_image < threshold) * 255

        return (255 - binary_image).astype(np.uint8)
