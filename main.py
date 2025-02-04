import pygame
import sys
import numpy as np
import cv2
import cairosvg
from keras.models import load_model
from pygame.locals import *

# Constants
WINDOWSIZEX = 1000  # Increased width for better spacing
WINDOWSIZEY = 750  # Increased height for better UI
BOUNDRYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
DARK_GREEN = (22, 64, 61)  # Background color
TEXT_COLOR = (229, 229, 229)  # Light gray for text
BOX_COLOR = (30, 30, 30)  # Dark gray for the drawing area
BORDER_COLOR = (200, 200, 200)  # Light gray border

IMAGESAVE = False

# Load Model
MODEL = load_model("/home/dell/Téléchargements/Handwritten-Digit-Recognition-main/bestmodel.h5")

LABELS = {0: "Zero", 1: "One",
          2: "Two", 3: "Three",
          4: "Four", 5: "Five",
          6: "Six", 7: "Seven",
          8: "Eight", 9: "Nine"}

# Initialize pygame
pygame.init()
FONT = pygame.font.Font(None, 36)  # Default font
HEADER_FONT = pygame.font.Font(None, 55)  # Bigger font for title
SCRIPT_FONT = pygame.font.Font(None, 80)  # Fancy script-like font for title

LOGO = pygame.image.load("logo-arsii-light.jpg")
LOGO = pygame.transform.scale(LOGO, (180, 90))

DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Handwritten Digit Recognition")

iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1
REDICT = True

# Clear screen function
def clear_screen():
    DISPLAYSURF.fill(WHITE)
    DISPLAYSURF.blit(LOGO, (50, 50))
    pygame.draw.rect(DISPLAYSURF, BORDER_COLOR, (50, 170, WINDOWSIZEX - 100, WINDOWSIZEY - 300), border_radius=10)  # Border
    pygame.draw.rect(DISPLAYSURF, BOX_COLOR, (55, 175, WINDOWSIZEX - 110, WINDOWSIZEY - 310), border_radius=10)  # Inner area
    text_instructions()

# Display text instructions
def text_instructions():
    title_surface = HEADER_FONT.render("Learn Build Evolve", True, BLACK)
    instruction_surface = FONT.render("Draw a digit inside the box and release mouse to predict", True, BLACK)
    clear_surface = FONT.render("Press 'C' to Clear, 'Q' to Quit", True, BLACK)

    DISPLAYSURF.blit(title_surface, (WINDOWSIZEX // 2 - title_surface.get_width() // 2, 50))
    DISPLAYSURF.blit(instruction_surface, (WINDOWSIZEX // 2 - instruction_surface.get_width() // 3, 90))
    DISPLAYSURF.blit(clear_surface, (WINDOWSIZEX // 2 - clear_surface.get_width() // 2, 120))

    if LOGO:
        DISPLAYSURF.blit(LOGO, (50, 50))  # Display logo at the top-left

# Initialize screen
clear_screen()

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            if 80 < xcord < 770 and 180 < ycord < 570:  # Restrict drawing to box
                pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
                number_xcord.append(xcord)
                number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if number_xcord and number_ycord:
                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)

                rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 80), min(770, number_xcord[-1] + BOUNDRYINC)
                rect_min_Y, rect_max_Y = max(number_ycord[0] - BOUNDRYINC, 180), min(number_ycord[-1] + BOUNDRYINC, 570)

                number_xcord = []
                number_ycord = []

                img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_Y:rect_max_Y].T.astype(np.float32)

                if IMAGESAVE:
                    cv2.imwrite("image.png", img_arr)
                    image_cnt += 1

                if REDICT:
                    image = cv2.resize(img_arr, (28, 28))
                    image = np.pad(image, ((10, 10), (10, 10)), mode='constant', constant_values=0)
                    image = cv2.resize(image, (28, 28))
                    image = image / 255.0

                    label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                    textSurface = FONT.render(label, True, RED, WHITE)
                    TextRectObj = textSurface.get_rect()
                    TextRectObj.center = ((rect_min_x + rect_max_x) // 2, rect_min_Y - 20)

                    DISPLAYSURF.blit(textSurface, TextRectObj)

        if event.type == KEYDOWN:
            if event.unicode == 'c':  # Press 'C' to clear screen
                clear_screen()
            if event.unicode == 'q':  # Press 'Q' to quit
                pygame.quit()
                sys.exit()

    copyright_font = pygame.font.Font(None, 30)
    name_surface = copyright_font.render("Bechir Karmeni | 5 February 2025", True, BLACK)
    date_surface = copyright_font.render("ESSTHS, Salle 4C", True, BLACK)
    DISPLAYSURF.blit(name_surface, (WINDOWSIZEX // 2 - name_surface.get_width() // 2, 630))
    DISPLAYSURF.blit(date_surface, (WINDOWSIZEX // 2 - date_surface.get_width() // 2, 660))
    pygame.display.flip()
