from working import ConcreteFingerRegionRecognizer
from second_imlementation import SecondImplementation
import cv2
import os
import numpy as np
import argparse
from copy import deepcopy
from abc import ABC
import matplotlib.pyplot as plt


def read_image(img_name: str):
    image = cv2.imread(img_name, 0)
    img_name = os.path.basename(img_name)
    print(f"working on image: {img_name}")
    return image


def show_image(image_1, algo_type: str):
    # Convert the images from BGR to RGB
    img1 = image_1
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    # Display the images
    axs.imshow(img1)
    axs.set_title(f"{algo_type} algorithm:")
    axs.axis("off")

    # Save the figure
    new_name = f"{algo_type}.png"
    print(f"Save img: {new_name}")
    fig.savefig(new_name)


def show_images(image_1, image_2, image_name, step: str):
    # Convert the images from BGR to RGB
    try:
        img1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    except:
        img1 = cv2.cvtColor(image_1.astype("float32"), cv2.COLOR_BGR2RGB)
    try:
        img2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    except:
        img2 = cv2.cvtColor(image_2.astype("float32"), cv2.COLOR_BGR2RGB)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Display the images
    axs[0].imshow(img1)
    axs[0].set_title(f"Working algorithm: step {step}")
    axs[0].axis("off")

    axs[1].imshow(img2)
    axs[1].set_title(f"Not working algorithm: step {step}")
    axs[1].axis("off")

    if np.array_equal(image_1, image_2):
        print("image1 and image2 are the same")
    else:
        print("image1 and image2 are different")
    # Save the figure
    new_name = image_name.replace(".png", "") + "_" + step + "_combined.png"
    print(f"image saved to: {new_name}")
    fig.savefig(new_name)


def client_code(abstract_class: ABC) -> None:
    """
    The client code calls the template method to execute the algorithm. Client
    code does not have to know the concrete class of an object it works with, as
    long as it works with objects through the interface of their base class.
    """
    return abstract_class.run()

def compare_step_by_step(image_name, image):
    greyscale_image1 = deepcopy(image)
    greyscale_image2 = deepcopy(image)
    working_alg1 = ConcreteFingerRegionRecognizer(greyscale_image1)
    not_working_alg1 = SecondImplementation(greyscale_image2)
    step1_working = working_alg1.first_step()
    step1_not_working = not_working_alg1.first_step()
    show_images(step1_working, step1_not_working, image_name, "1")

    step2_working = working_alg1.second_step()
    step2_not_working = not_working_alg1.second_step()
    show_images(step2_working, step2_not_working, image_name, "2")

    step3_working = working_alg1.third_step()
    step3_not_working = not_working_alg1.third_step()
    show_images(step3_working, step3_not_working, image_name, "3")

    step4_working = working_alg1.fourth_step()
    step4_not_working = not_working_alg1.fourth_step()
    show_images(step4_working, step4_not_working, image_name, "4")

def replace_steps(obj_to_run, obj_to_take, steps):
    obj_to_run.first_step()
    obj_to_take.first_step()
    if 1 in steps:
        if isinstance(obj_to_run, ConcreteFingerRegionRecognizer):
            print("obj_to_run is an instance of ConcreteFingerRegionRecognizer")
            obj_to_run.normalized_img = obj_to_take.normim
        else:
            print("obj_to_run is an instance of SecondImplementation")
            obj_to_run.normim = obj_to_run.normalized_img

    obj_to_run.second_step()
    obj_to_take.second_step()
    if 2 in steps:
        obj_to_run.orientation = obj_to_take.orientation

    obj_to_run.third_step()
    obj_to_take.third_step()
    if 3 in steps:
        obj_to_run.frequency = obj_to_take.frequency

    obj_to_run.fourth_step()
    obj_to_take.fourth_step()
    if 4 in steps:
        obj_to_run.gabor_img = obj_to_take.gabor_img
    obj_to_run.fifth_step()
    obj_to_take.fifth_step()
    result = obj_to_run.run_second_phase()
    show_image(result, "replace_steps_" + "_".join(map(str, steps)))


if __name__ == "__main__":
    input_path = "./all_png_files/M89_f0126_10.png"
    output_path = "./all_png_files_out/"
    not_working_out = "./not_working/"

    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument(
        "-i",
        dest="img_name",
        type=str,
        help="Path to the input images",
        default=input_path,
    )

    args = parser.parse_args()

    img_name = args.img_name
    if not img_name:
        img_name = input_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(not_working_out):
        os.makedirs(not_working_out)

    # greyscale_image3 = deepcopy(greyscale_image1)

    output_path = os.path.join(output_path, img_name)
    nw_output_path = os.path.join(not_working_out, img_name)

    # greyscale_image1 = read_image(f"./all_png_files/{img_name}")
    greyscale_image1 = read_image(input_path)
    greyscale_image2 = deepcopy(greyscale_image1)
    working_alg1 = ConcreteFingerRegionRecognizer(greyscale_image1)
    not_working_alg1 = SecondImplementation(greyscale_image2)
    # working_img = client_code(working_alg1)
    # not_working_img = client_code(not_working_alg1)
    # show_image(working_img, "working")
    # show_image(not_working_img, "not_working")
    img_base_name = os.path.basename(img_name)
    # show_images(working_img, not_working_img, img_base_name, "final")
    # compare_step_by_step(img_base_name, greyscale_image1)
    replace_steps(not_working_alg1, working_alg1, [1, 2, 3, 4], prefix="replace_steps_")


    # if os.path.isdir(img_name):
    #     for img_name in os.listdir("./all_png_files"):
    #         for i in range(1, 6):
    #             if i in [2, 3]:
    #                 continue
    #             try:
    #                 print(f"Running image: {img_name} with step: {i}")
    #                 greyscale_image1 = read_image(f"./all_png_files/{img_name}")
    #                 greyscale_image2 = deepcopy(greyscale_image1)
    #                 greyscale_image4 = deepcopy(greyscale_image1)
    #                 working_alg = ConcreteFingerRegionRecognizer()
    #                 not_working_alg = SecondImplementation()
    #                 working_alg1 = ConcreteFingerRegionRecognizer()
    #
    #                 first_step_working = working_alg.first_step(greyscale_image1)
    #                 first_step_not_working = not_working_alg.first_step(greyscale_image2)
    #                 if i >= 1:
    #                     # working_alg.normalized_img = not_working_alg.normim
    #                     not_working_alg.normim = working_alg.normalized_img
    #
    #                 third_step_working = working_alg.second_step()
    #                 third_step_not_working = not_working_alg.second_step()
    #
    #                 fourth_step_working = working_alg.third_step()
    #                 fourth_step_not_working = not_working_alg.third_step()
    #                 if i >= 4:
    #                     # working_alg.frequency = not_working_alg.freq
    #                     not_working_alg.freq = working_alg.frequency
    #
    #                 fifth_step_working = working_alg.fifth_step()
    #                 fifth_step_not_working = not_working_alg.fifth_step()
    #                 if i >= 5:
    #                     # working_alg.gabor_img = not_working_alg.gabor_img
    #                     not_working_alg.gabor_img = working_alg.gabor_img
    #
    #                 full_working_run = working_alg1.run(greyscale_image4)
    #                 image_not_working_final = working_alg.run_second_phase(greyscale_image1)
    #                 image_not_working_fixed = not_working_alg.run_second_phase(
    #                     greyscale_image2
    #                 )
    #                 show_images(
    #                     full_working_run,
    #                     image_not_working_fixed,
    #                     f"reversed-{img_name}-step{list(range(1,i+1))}",
    #                 )
    #             except Exception as e:
    #                 print(f"Failed running the image: {img_name} with step: {i}")
    #                 print(f"Error: {e}")
    #                 continue
    #         break



