import cv2
import numpy as np


class GrabcutSegmentation:
    def grabcut_segmentation(img):
        original_image = cv2.imread(str(img))
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        binarized_image = cv2.adaptiveThreshold(
            gray_image,
            maxValue=1,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=13,
            C=7,
        )

        cv2.setRNGSeed(0)
        number_of_iterations = 6

        # Define boundary rectangle containing the foreground object
        height, width, _ = original_image.shape
        left_margin_proportion = 0.3
        right_margin_proportion = 0.3
        up_margin_proportion = 0.1
        down_margin_proportion = 0.1

        boundary_rectangle = (
            int(width * left_margin_proportion),
            int(height * up_margin_proportion),
            int(width * (1 - right_margin_proportion)),
            int(height * (1 - down_margin_proportion)),
        )

        # Initialize mask image
        mask = np.zeros((height, width), np.uint8)

        # Arrays used by the algorithm internally
        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)

        # Initialize the mask with known information
        initial_mask = binarized_image.copy()
        # show_image(255 * initial_mask, "Initial mask")

        mask = np.zeros((height, width), np.uint8)
        mask[:] = cv2.GC_PR_BGD
        mask[initial_mask == 0] = cv2.GC_FGD

        # Arrays used by the algorithm internally
        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            original_image,
            mask,
            boundary_rectangle,
            background_model,
            foreground_model,
            number_of_iterations,
            cv2.GC_INIT_WITH_MASK,
        )

        grabcut_mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype(
            "uint8"
        )

        grabcut_image = original_image.copy() * grabcut_mask[:, :, np.newaxis]
        return grabcut_image
