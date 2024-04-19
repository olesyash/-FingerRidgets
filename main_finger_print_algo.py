from working import ConcreteFingerRegionRecognizer
from second_imlementation import SecondImplementation
import cv2
import os
import argparse
from copy import deepcopy


if __name__ == '__main__':
    input_path = './all_png_files/M89_f0104_08.png'
    output_path = './all_png_files_out/'
    not_working_out = './not_working/'

    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('-i', dest="img_name", type=str, help='Path to the input images', default=input_path)


    args = parser.parse_args()

    img_name = args.img_name
    if not img_name:
        img_name = input_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

       
    greyscale_image1 = cv2.imread(img_name, 0)
    greyscale_image2 = deepcopy(greyscale_image1)
    print(img_name)
    block_size = int(16)

    working_alg = ConcreteFingerRegionRecognizer()
    not_working_alg = SecondImplementation()

    normalized_img1 = working_alg.normalize_image(greyscale_image1)
    original_img = working_alg.unnormilize_image(normalized_img1)
    normalized_img2 = not_working_alg.normalize_image(greyscale_image2)
    # # Compare normalization results
    cv2.imwrite("normilized1.png", original_img)
    cv2.imwrite("normilized2.png", normalized_img2)
    # # Compare segmentation results
    mask1, img_norm_masked1 = working_alg.segment_image(normalized_img1)
    segmented_img2, normim2, mask2 = not_working_alg.segment_image(normalized_img2, 16)
    original_img = working_alg.unnormilize_image(img_norm_masked1)
    cv2.imwrite("segmented1.png", original_img)
    cv2.imwrite("segmented2.png", segmented_img2)

    orientation1 = working_alg.lines_orientation(img_norm_masked1)
    angles2, orientation2 = not_working_alg.lines_orientation(normalized_img2, mask2, segmented_img2, 16, smoth=False)
    # # original_img = working_alg.unnormilize_image(orientation1)
    cv2.imwrite("orientation1.png", orientation1)
    cv2.imwrite("orientation2.png", orientation2)

    frequency1 = working_alg.calculate_ridge_frequencies(img_norm_masked1, orientation1, mask1)
    # # find the overall frequency of ridges in Wavelet Domain
    frequency2 = not_working_alg.calculate_ridge_frequencies(normim2, mask2, angles2, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
    # # Compare frequency results
    print(len(frequency1))
    print(len(frequency2))

    # # create gabor filter and do the actual filtering
    gabor_img1 = working_alg.apply_gabor_filter(img_norm_masked1, frequency1, orientation1)
    gabor_img2 = not_working_alg.apply_gabor_filter(normim2, angles2, frequency2)
    cv2.imwrite("gabor1.png", gabor_img1)
    cv2.imwrite("gabor2.png", gabor_img2)

    skeletonized_img1 = working_alg.skeletonize(gabor_img1)
    skeletonized_img2 = not_working_alg.skeletonize(gabor_img2)
    cv2.imwrite("skeletonized1.png", skeletonized_img1)
    cv2.imwrite("skeletonized2.png", skeletonized_img2)

    # output_image = working_alg.count_fingerprint_ridges(greyscale_image)
    # cv2.imwrite(os.path.join(output_path, os.path.basename(img_name)), output_image)

    # not_working_alg = SecondImplementation()
    # output_image_not_working = not_working_alg.count_fingerprint_ridges(greyscale_image)
    # cv2.imwrite(not_working_out + img_name, output_image)

