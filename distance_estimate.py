"""
Sample input
4/3
84
7.17
4.48
10.6
-54.8

aspect ratio
field of vision (degrees)
height of image (pixel)
lower ordinate from top of the image (pixel)
camera height (meter)
gimbal pitch (degrees)
drone pitch (degrees)
"""

import fractions
import math

with open('input') as input_file:
  aspect_ratio: float = float(fractions.Fraction(input_file.readline()))
  fov: float = math.radians(float(input_file.readline()))
  H: float = float(input_file.readline())
  h: float = float(input_file.readline())
  Hc: float = float(input_file.readline())
  gimbal_pitch: float = math.radians(-float(input_file.readline()))
  drone_pitch: float = math.radians(float(input_file.readline()))

pitch = gimbal_pitch + drone_pitch # TODO: See if this is right
lens_scaling = math.hypot(1, aspect_ratio) / math.tan(fov / 2)
horizon = (H / 2) * (1 - math.tan(pitch) * lens_scaling)
scaled_height = lens_scaling * (H / 2) / (h - horizon)
distance = Hc * scaled_height / (math.cos(pitch) ** 2) - Hc * math.tan(pitch)
print(f'Distance: {distance}')
