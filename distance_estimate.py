import fractions
import math

aspect_ratio: float = float(fractions.Fraction(input('aspect ratio: ')))
fov: float = math.radians(float(input('field of vision (degrees): ')))
h: float = float(input('distance from foot to horizon (unit): '))
a: float = float(input('height of object (unit): '))
Hc: float = float(input('camera height (meter): '))
theta: float = math.radians(float(input('theta (degrees): ')))

lens_scaling = math.hypot(1, aspect_ratio) / math.tan(fov / 2)
scaled_height = lens_scaling * a / h
distance = 1 / (math.cos(theta) ** 2) * Hc * scaled_height - Hc * math.tan(theta)
print(distance)
