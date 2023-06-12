import pyautogui
import numpy as np
import pywinctl as pwc
from PIL import Image
from mss import mss, tools

import time


    
####### THIS SECTION SHOULD RUN ONCE #########
# Get the application window
window_name = 'rscplus (RSC Preservation; friqi)'
window = pwc.getWindowsWithTitle(window_name)[0]

# Focus on the window
window.activate()

# Define the region of the window (x, y, width, height)
SEARCH_REGION = (window.left, window.top, window.width, window.height)

sct = mss()
MONITOR = {"top": SEARCH_REGION[1], "left": SEARCH_REGION[0], "width": SEARCH_REGION[2], "height": SEARCH_REGION[3]}
##############################################

def scale(window_width, window_height, screenshot_width,
          screenshot_height, x, y):
    return (x * window_width / screenshot_width,
            y * window_height / screenshot_height)

def cut_tree():
    window.activate()
    # Define the color you want to detect (in RGB format)
    wood_of_tree = (145, 80, 39)
    screenshot = get_screenshot()

    # Convert the screenshot to a numpy array
    image = np.array(screenshot)
    
    BROWN_COLOR_RANGE = {
        "R": (120, 150),
        "G": (50, 70),
        "B": (10, 30)
        # "R": (150, 200),
        # "G": (100, 150),
        # "B": (50, 100)
    }

    COLOR_RANGE = BROWN_COLOR_RANGE
    mask = (
        (image[:,:,0] >= COLOR_RANGE["R"][0]) & (image[:,:,0] <= COLOR_RANGE["R"][1]) &
        (image[:,:,1] >= COLOR_RANGE["G"][0]) & (image[:,:,1] <= COLOR_RANGE["G"][1]) &
        (image[:,:,2] >= COLOR_RANGE["B"][0]) & (image[:,:,2] <= COLOR_RANGE["B"][1])
    )
    
    
    # Search for the target color
    indices = np.where(mask)
    
    # If the color is found, move the mouse to the first occurrence and click
    if indices[0].size > 0:
        y, x = indices[0][0], indices[1][0]
        window_width, window_height = SEARCH_REGION[2], SEARCH_REGION[3]
        # screenshot_width, screenshot_height = image.shape[0], image.shape[1]
        screenshot_width, screenshot_height = image.shape[1], image.shape[0]
        
        x, y = scale(window_width, window_height, screenshot_width, screenshot_height, x, y)

        screen_x = x + SEARCH_REGION[0]
        screen_y = y + SEARCH_REGION[1]
        print('screen x / y', screen_x, screen_y)
        pyautogui.moveTo(screen_x, screen_y)
        pyautogui.click()
        print(f"Color found and clicked at ({screen_x}, {screen_y})")
    else:
        print("Color not found")
    return


def get_screenshot():
    screenshot = sct.grab(MONITOR)
    
    # for debugging - to print out output
    output = "sct-{top}x{left}_{width}x{height}.png".format(**MONITOR)
    tools.to_png(screenshot.rgb, screenshot.size, output="/tmp/{}".format(output))
    return screenshot
    
    
    
if __name__ == '__main__':
    count = 0
    while count < 60:
        cut_tree()
        time.sleep(3)
        count += 1
