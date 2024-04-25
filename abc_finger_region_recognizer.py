from abc import ABC, abstractmethod
import numpy as np
import cv2
from skimage.morphology import skeletonize as ski_skeletonize
import os
from scipy import ndimage
from scipy.ndimage import rotate, grey_dilation


class FingerRegionRecognizerTemplate(ABC):
    """
    The FingerReGionRecognizer template has few implemetned functions and it declares the operations that all concrete FingerRegionRecognizers
    must implement.
    """
    @abstractmethod
    def normalize_image(self) -> str:
        pass

    @abstractmethod
    def segment_image(self) -> str:
        pass

    @abstractmethod
    def lines_orientation(self) -> str:
        pass

    @abstractmethod
    def calculate_ridge_frequencies(self) -> str:
        pass

    # @abstractmethod
    # def apply_gabor_filter(self) -> str:
    #     pass

    def apply_gabor_filter(self, image, frequency, orientation, threshold=-2, kx=0.65, ky=0.65, angle_increment=3):
        image = image.astype(np.float64)
        rows, cols = image.shape
        filtered_image = np.zeros((rows, cols))

        non_zero_freqs = frequency.ravel()[frequency.ravel() > 0]
        rounded_freqs = np.round(non_zero_freqs * 100) / 100
        unique_freqs = np.unique(rounded_freqs)

        sigma_x = 1 / unique_freqs[0] * kx
        sigma_y = 1 / unique_freqs[0] * ky
        filter_size = np.int_(np.round(3 * np.max([sigma_x, sigma_y])))

        x, y = np.meshgrid(np.arange(-filter_size, filter_size + 1), np.arange(-filter_size, filter_size + 1))
        exponent = ((x / sigma_x) ** 2 + (y / sigma_y) ** 2) / 2
        reference_filter = np.exp(-exponent) * np.cos(2 * np.pi * unique_freqs[0] * x)
        angle_range = int(180 / angle_increment)
        gabor_filters = np.array(
            [rotate(reference_filter, -(o * angle_increment + 90), reshape=False) for o in range(angle_range)])

        max_size = filter_size
        valid_rows, valid_cols = np.where(frequency > 0)
        valid_indices = np.where((valid_rows > max_size) & (valid_rows < rows - max_size) & (valid_cols > max_size) & (
                valid_cols < cols - max_size))[0]
        max_orient_index = int(np.round(180 / angle_increment))
        orient_index = np.round(orientation / np.pi * 180 / angle_increment).astype(np.int32)
        orient_index[orient_index < 1] += max_orient_index
        orient_index[orient_index > max_orient_index] -= max_orient_index

        for k in valid_indices:
            r = valid_rows[k]
            c = valid_cols[k]
            img_block = image[r - filter_size:r + filter_size + 1, c - filter_size:c + filter_size + 1]
            filtered_image[r, c] = np.sum(img_block * gabor_filters[orient_index[r, c] - 1])

        binary_image = (filtered_image < threshold) * 255

        return (255 - binary_image).astype(np.uint8)

    @abstractmethod
    def calculate_ridge_frequencies(self) -> str:
        pass


    def template_method(self) -> None:
        """
        The FingerRegionRecognizer template method defines the skeleton of an algorithm.
        """
        self.normalize_image()
        self.segment_image()
        self.lines_orientation()
        self.calculate_ridge_frequencies()
        self.apply_gabor_filter()

    def skeletonize(self, img):
        binary_image = np.zeros_like(img)
        binary_image[img == 0] = 1.0

        skeleton = ski_skeletonize(binary_image)

        output_img = (1 - skeleton) * 255.0
        return output_img.astype(np.uint8)
    
    def detect_minutiae(self, image, row, col):
        if image[row][col] == 1:
            kernel = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
            values = []
            for k, l in kernel:
                values.append(image[row + l][col + k])
            crossings = sum(abs(values[k] - values[k + 1]) for k in range(len(values) - 1)) // 2
            if crossings == 1:
                return "termination"
            if crossings == 3:
                return "bifurcation"
        return "none"
    
    def calculate_minutiae_weights(self, image):
        binary_image = (image == 0).astype(np.int8)
        minutiae_weights_array = np.zeros_like(image, dtype=np.float_)

        for col in range(1, image.shape[1] - 1):
            for row in range(1, image.shape[0] - 1):
                minutiae = self.detect_minutiae(binary_image, row, col)
                if minutiae == "bifurcation":
                    minutiae_weights_array[row, col] += 1.0
                elif minutiae == "termination":
                    minutiae_weights_array[row, col] += 2.0

        return minutiae_weights_array


    def count_lines(self, image):
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded_image = cv2.erode(image, kernel)
        binary_image = cv2.threshold(eroded_image, 127, 255, cv2.THRESH_BINARY)[1]
        main_diagonal = self.count_diagonal_lines(binary_image)
        secondary_diagonal = self.count_diagonal_lines(np.fliplr(binary_image))
        return (secondary_diagonal, "secondary_diagonal") if main_diagonal <= secondary_diagonal else (
            main_diagonal, "main_diagonal")


    def count_diagonal_lines(self, image):
        is_white = True
        counter = 0
        for i in range(image.shape[0]):
            if image[i][i] == 0 and is_white:
                counter += 1
                is_white = False
            if image[i][i] == 255:
                is_white = True
        return counter


    def draw_diagonal(self, image, start_i, start_j, end_i, end_j, line_count, block_size, text_y, text_x, color):
        cv2.line(image, (start_j, start_i), (end_j, end_i), (0, 0, 255), 1)
        cv2.putText(image, f' {line_count}', (text_y, text_x - block_size), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1)
        cv2.putText(image, "ridges", (text_y, text_x), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1)


    def get_best_region(self, thin_image, minutiae_weights_image, block_size, mask):
        best_region = None
        block_minutiaes_weight = float('inf')
        rows, cols = thin_image.shape
        shape_block = block_size * 8
        window = (shape_block // block_size) - 1
        for col in range(1, cols - window):
            for row in range(1, rows - window):
                mask_slice = mask[col * block_size:col * block_size + shape_block,
                            row * block_size:row * block_size + shape_block]
                mask_flag = np.sum(mask_slice)
                if mask_flag == shape_block * shape_block:
                    number_vision_problems = np.sum(minutiae_weights_image[col * block_size:col * block_size + shape_block,
                                                    row * block_size:row * block_size + shape_block])
                    if number_vision_problems <= block_minutiaes_weight:
                        block_minutiaes_weight = number_vision_problems
                        best_region = [col * block_size, col * block_size + shape_block, row * block_size,
                                    row * block_size + shape_block]
        if best_region:
            return best_region


    def draw_ridges_count_on_region(self, region, input_image, thin_image, block_size):
        output_image = cv2.cvtColor(input_image.copy(), cv2.COLOR_GRAY2RGB)
        if region is None:
            return output_image
        region_copy = (thin_image[region[0]: region[1], region[2]: region[3]]).copy()
        line_count, line_type = self.count_lines(region_copy)
        text_y = (region[3] - region[2]) // 2 + region[2]
        if line_type == "main_diagonal":
            text_x = (region[1] - region[0]) // 3 + region[0]
            self.draw_diagonal(output_image, region[0], region[2], region[1], region[3], line_count, block_size, text_y, text_x,
                        (0, 255, 255))
        elif line_type == "secondary_diagonal":
            text_x = 2 * (region[1] - region[0]) // 3 + region[0]
            self.draw_diagonal(output_image, region[0], region[3], region[1], region[2], line_count, block_size, text_y, text_x,
                        (0, 255, 255))
        cv2.rectangle(output_image, (region[2], region[0]), (region[3], region[1]), (0, 0, 255), 1)
        return output_image
