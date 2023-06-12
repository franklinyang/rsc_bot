from bot import scale

import pyautogui
import numpy as np
import pywinctl as pwc
import cv2
from mss import mss, tools

import os
from collections import namedtuple

# Get the application WINDOW
WINDOW = pwc.getWindowsWithTitle('rscplus (RSC Preservation; friqi)')[0]

# Define the region of the WINDOW (x, y, width, height)
SEARCH_REGION = (WINDOW.left, WINDOW.top, WINDOW.width, WINDOW.height)

Size = namedtuple('Size', ['height', 'width'])


def load_trained_images(directory):
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return [cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE) for filename in image_files]


def capture_screenshot(region):
    sct = mss()
    monitor = {"top": region[1], "left": region[0], "width": region[2], "height": region[3]}
    # for debugging - to print out output
    output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)
    screenshot = sct.grab(monitor)

    screen = np.array(screenshot)
    # Resize to match region dimensions
    screen = cv2.resize(screen, (region[2], region[3]))
    
    return screen

# def find_and_click_trained_image(screen, trained_images, SEARCH_REGION, scales=[0.8, 1.0, 1.2]):
#     gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
# 
#     for template in trained_images:
#         res = cv2.matchTemplate(gray_screen, template, cv2.TM_CCOEFF_NORMED)
#         threshold = 0.8
#         loc = np.where(res >= threshold)
# 
#         WINDOW_width, WINDOW_height = SEARCH_REGION[2], SEARCH_REGION[3]
#         print('window:', WINDOW_width, WINDOW_height)
#         screenshot_height, screenshot_width = res.shape
#         print('screenshot:', res.shape)
# 
#         for pt in zip(*loc[::-1]):
#             print('found:', pt[0], pt[1])
#             x, y = scale(WINDOW_width, WINDOW_height, screenshot_width, screenshot_height, pt[0], pt[1])
#             screen_x = x + SEARCH_REGION[0]
#             screen_y = y + SEARCH_REGION[1]
#             pyautogui.moveTo(screen_x, screen_y)
#             print(screen_x, screen_y)
#             pyautogui.click()
#             print(f"Image found and clicked at ({screen_x}, {screen_y})")
#             return True
#     return False
def find_and_click_trained_image(screen, trained_images, search_region, scales=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    
    for template in trained_images:
        # Loop over the scales
        for scale in scales:
            # Resize the template according to the scale
            resized_template = cv2.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))
            res = cv2.matchTemplate(gray_screen, resized_template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)
            
            for pt in zip(*loc[::-1]):
                print('found at scale:', scale)
                screen_x = pt[0] + search_region[0]
                screen_y = pt[1] + search_region[1]
                pyautogui.moveTo(screen_x, screen_y)
                pyautogui.click()
                print(f"Image found and clicked at ({screen_x}, {screen_y})")
                return True
    return False


def main():

    # Load the trained images
    trained_images = load_trained_images('training_images')
    print(trained_images)

    # Focus on the WINDOW
    WINDOW.activate()

    screenshot = capture_screenshot(SEARCH_REGION)
    find_and_click_trained_image(screenshot, trained_images, SEARCH_REGION)
    # Continuously capture screenshots and search for the trained images
    # while True:
    #     screenshot = capture_screenshot(SEARCH_REGION)
    #     screen = np.array(screenshot)
    #     print(screen.shape)
    #     print('looking')
    #     if find_and_click_trained_image(screen, trained_images, SEARCH_REGION):
    #         break

if __name__ == "__main__":
    main()
