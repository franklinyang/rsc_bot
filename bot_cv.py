from bot import scale

import pyautogui
import numpy as np
import pywinctl as pwc
import cv2
from mss import mss, tools

import os
from collections import namedtuple

# Get the application WINDOW
WINDOW_TITLE = 'rscplus (RSC Preservation; friqi)'
WINDOW = pwc.getWindowsWithTitle(WINDOW_TITLE)[0]

# Define the region of the WINDOW (x, y, width, height)
SEARCH_REGION = (WINDOW.left, WINDOW.top, WINDOW.width, WINDOW.height)

Size = namedtuple('Size', ['height', 'width'])


def load_trained_images(directory):
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return [cv2.imread(os.path.join(directory, filename)) for filename in image_files]
    # return [cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE) for filename in image_files]



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

# models: [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
def find_and_click_trained_image(screen, trained_images, search_region, scales=[0.4, 0.5, 0.6, 0.7]):
    screen = screen[:, :, :3].astype(np.uint8)
    
    for template in trained_images:
        template = template.astype(np.uint8)
        # Loop over the scales
        for scale in scales:
            # Resize the template according to the scale
            resized_template = cv2.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))
            resized_template = resized_template.astype(np.uint8)

            res = cv2.matchTemplate(screen, resized_template, cv2.TM_CCOEFF_NORMED)
            debug_img(res)
            threshold = 0.4
            loc = np.where(res >= threshold)
            
            for pt in zip(*loc[::-1]):
                print('found at scale:', scale, 'threshold:', threshold)
                screen_x = pt[0] + search_region[0]
                screen_y = pt[1] + search_region[1]
                pyautogui.moveTo(screen_x, screen_y)
                pyautogui.click()
                print(f"Image found and clicked at ({screen_x}, {screen_y})")
                return True
    return False


def feature_matching_old(screen, template, scales=[0.4, 0.5, 0.6, 0.7]):
    screen = screen[:, :, :3].astype(np.uint8)

    scale = 0.5
    resized_template = cv2.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))
    resized_template = resized_template.astype(np.uint8)

    # ORB detector
    orb = cv2.ORB_create()
    
    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(resized_template, None)
    kp2, des2 = orb.detectAndCompute(screen, None)
    
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    for match in matches:
        # Get the coordinates of the match
        x, y = map(int, kp2[match.trainIdx].pt)
        
        # Draw rectangle around the match
        h, w = resized_template.shape[:2]
        cv2.rectangle(screen, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Show the image with rectangles
    cv2.imshow('matches', screen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # if matches:
    #     # Draw first match
    #     img_matches = cv2.drawMatches(template, kp1, screen, kp2, matches[:1], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #     cv2.imshow('matches', img_matches)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     
    #     # Get the coordinates of the first match
    #     x, y = kp2[matches[0].trainIdx].pt
    #     
    #     # Perform the click action at the given x, y coordinates
    #     pyautogui.moveTo(x, y)
    #     pyautogui.click()
    #     print(f"Image found and clicked at ({x}, {y})")


def feature_matching(screen, template, max_distance=60, max_matches=5):
    screen = screen[:, :, :3].astype(np.uint8)

    scale = 0.5
    resized_template = cv2.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))
    resized_template = resized_template.astype(np.uint8)

    # ORB detector
    orb = cv2.ORB_create()
    
    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(resized_template, None)
    kp2, des2 = orb.detectAndCompute(screen, None)
    
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    

    img_template_keypoints = cv2.drawKeypoints(resized_template, kp1, None, color=(0, 255, 0))
    cv2.imshow('Template Keypoints', img_template_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Visualize keypoints on the screen image
    img_screen_keypoints = cv2.drawKeypoints(screen, kp2, None, color=(0, 255, 0))
    cv2.imshow('Screen Keypoints', img_screen_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Counter for matches
    match_count = 0
    
    # Draw matches with distance less than max_distance
    for match in matches:
        if match.distance < max_distance:
            # Get the coordinates of the match
            x, y = map(int, kp2[match.trainIdx].pt)
            
            # Draw rectangle around the match
            h, w = resized_template.shape[:2]
            cv2.rectangle(screen, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            match_count += 1
            if match_count >= max_matches:
                break
    
    # Show the image with rectangles
    cv2.imshow('matches', screen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def debug_img(nparray):
    cv2.imshow(WINDOW_TITLE, nparray)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image


def main():

    # Load the trained images
    trained_images = load_trained_images('training_images')

    # Focus on the WINDOW
    WINDOW.activate()

    screenshot = capture_screenshot(SEARCH_REGION)
    feature_matching(screenshot, trained_images[0])
    # find_and_click_trained_image(screenshot, trained_images, SEARCH_REGION)
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
