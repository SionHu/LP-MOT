"""
Sample input
4/3
84
3.7
7.17
15.4
-12.7
"""

import fractions
import math

aspect_ratio: float = float(fractions.Fraction(input('aspect ratio: ')))
fov: float = math.radians(float(input('field of vision (degrees): ')))
h: float = float(input('distance from foot to horizon (pixel): '))
H: float = float(input('height of image (pixel): '))
Hc: float = float(input('camera height (meter): '))
theta: float = math.radians(-float(input('gimbal pitch (degrees): ')))

lens_scaling = math.hypot(1, aspect_ratio) / math.tan(fov / 2)
scaled_height = lens_scaling * (H / 2) / h
distance = Hc * scaled_height / (math.cos(theta) ** 2) - Hc * math.tan(theta)
print(f'Distance: {distance}')
