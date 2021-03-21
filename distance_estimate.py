import math

aspect_ratio: float = float(input('aspect ratio: '))
fov: float = float(input('field of vision: '))
h: float = float(input('distance from foot to horizon: '))
a: float = float(input('height of object: '))
Hc: float = float(input('camera height: '))

lens_scaling = math.hypot(1, aspect_ratio) / math.tan(fov / 2)
scaled_height = lens_scaling * a / h
distance = 1 / (math.cos(theta) ** 2) * Hc * scaled_height - Hc * tan(theta)
print(distance)
