import pygame
import time
import numpy as np
from cartpole_env import CustomCartPoleEnv

# Create the custom CartPole environment
env = CustomCartPoleEnv()

# Initialize Pygame
pygame.init()

# Set up the screen dimensions
screen_width = 600
screen_height = 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("CART POLE")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Define function to draw cartpole
def draw_cartpole(screen, state):
    # Cart parameters
    cart_width = 50
    cart_height = 30
    
    # Cart position
    cart_x = int(state[0] * screen_width / 4) + screen_width // 2 - cart_width // 2
    cart_y = screen_height // 2 - cart_height // 2
    
    # Pole parameters
    pole_length = 100
    pole_angle = state[2]
    
    # Fill the background with a specific color
    screen.fill((144, 238, 144))  # LightSkyBlue color
    
    # Calculate pole end coordinates
    pole_end_x = cart_x + cart_width // 2 + pole_length * np.sin(pole_angle)
    pole_end_y = cart_y + cart_height // 2 - pole_length * np.cos(pole_angle)
    
    # Draw cart rectangle
    pygame.draw.rect(screen, BLACK, (cart_x, cart_y, cart_width, cart_height))
    
    # Draw cart base
    pygame.draw.line(screen, BLACK, (0, cart_y + cart_height // 2), (screen_width, cart_y + cart_height // 2), 2)
    
    # Draw pole with gold color and border
    pygame.draw.line(screen, (0, 0, 139), (cart_x + cart_width // 2, cart_y + cart_height // 2), (pole_end_x, pole_end_y), 6)

# Reset the environment
obs = env.reset()

# Set the duration of the game in seconds
game_duration = 10  # 10 seconds

start_time = time.time()
running = True
while running:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Take a random action
    action = env.action_space.sample()

    # Perform the action in the environment
    obs, reward, done, _ = env.step(action)

    # Draw the CartPole
    draw_cartpole(screen, obs)

    # Update the display
    pygame.display.flip()

    # Check if the game duration has elapsed
    if time.time() - start_time >= game_duration:
        running = False

    # Slow down the loop to see the animation
    time.sleep(0.02)

    if done:
        obs = env.reset()

# Close the environment
env.close()
pygame.quit()
print("GAME OVER")