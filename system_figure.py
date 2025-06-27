import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Polygon, Wedge, FancyArrowPatch, FancyArrow

fig, ax = plt.subplots(figsize=(6, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Lighting
light_box = Polygon([[4.7, 19], [5.3, 19], [5.2, 18.5], [4.8, 18.5]],
                    closed=True, facecolor='lightgray', edgecolor='black')
ax.add_patch(light_box)
# ax.text(5, 19.2, r'$Light$', ha='center', fontsize=10)

for dx in [-0.15, -0.07, 0, 0.07, 0.15]:
    ax.plot([5 + dx, 5], [18.5, 14.6], color='blue', lw=1)

# Platform (trapezoid)
platform_bottom_y = 14
platform_top_y = 15.2
platform_bottom_left = 2.8
platform_bottom_right = 7.2
platform_top_left = 3.1
platform_top_right = 6.9
platform = Polygon([
    [platform_bottom_left, platform_bottom_y],
    [platform_bottom_right, platform_bottom_y],
    [platform_top_right, platform_top_y],
    [platform_top_left, platform_top_y]
], closed=True, facecolor='lightgray', edgecolor='black')
ax.add_patch(platform)

# X-direction double-sided arrow
x_arrow = FancyArrowPatch((6, 13.6), (7, 13.6), arrowstyle='<->', mutation_scale=7, color='black')
ax.add_patch(x_arrow)
# ax.text(5, 14.3, 'X', ha='center')

# Z-direction double-sided arrow (vertical, center-left)
# z_arrow = FancyArrowPatch(
#     (6.9, 15.5),  # bottom point
#     (6.9, 17),  # top point
#     arrowstyle='<->', mutation_scale=7, color='black'
# )
# ax.add_patch(z_arrow)

# Calculate vector along right platform edge (top right to bottom right)
x_bottom, y_bottom = platform_bottom_right, platform_bottom_y
x_top, y_top = platform_top_right, platform_top_y

# Direction vector along platform edge (top to bottom)
dx = x_top - x_bottom
dy = y_top - y_bottom

# Normalize vector
length = (dx**2 + dy**2)**0.5
dx_norm = dx / length
dy_norm = dy / length

# Arrow length
arrow_length = 1.0  # adjust length as you want

# Arrow start point: slightly offset from bottom right corner (outwards)
offset = 0.3
move_x = 0.6
move_y = 0.3
x_start = x_bottom + move_x + offset * (-dy_norm)  # perpendicular offset for clarity
y_start = y_bottom + move_y + offset * dx_norm

# Arrow end point: start point + arrow_length along edge direction
x_end = x_start + arrow_length * dx_norm
y_end = y_start + arrow_length * dy_norm

# Draw double-headed arrow parallel to platform side (right edge)
y_arrow = FancyArrowPatch(
    (x_start, y_start),
    (x_end, y_end),
    arrowstyle='<->', mutation_scale=7, color='black')
ax.add_patch(y_arrow)
# ax.text(7.8, 14.6, 'Y', va='center')


# Sample (trapezoid with same slope, centered)
platform_height = platform_top_y - platform_bottom_y
platform_slope = (platform_bottom_right - platform_top_right) / platform_height

sample_height = 0.5
sample_bottom_y = platform_bottom_y + (platform_height - sample_height) / 2
sample_top_y = sample_bottom_y + sample_height

sample_bottom_width = 0.6 * (platform_bottom_right - platform_bottom_left)
sample_top_width = sample_bottom_width - 2 * platform_slope * sample_height

x_center = 5
sample_bottom_left = x_center - sample_bottom_width / 2
sample_bottom_right = x_center + sample_bottom_width / 2
sample_top_left = x_center - sample_top_width / 2
sample_top_right = x_center + sample_top_width / 2

sample = Polygon([
    [sample_bottom_left, sample_bottom_y],
    [sample_bottom_right, sample_bottom_y],
    [sample_top_right, sample_top_y],
    [sample_top_left, sample_top_y]
], closed=True, facecolor='white', edgecolor='black')
ax.add_patch(sample)
# ax.text(x_center, sample_top_y + 0.1, 'Sample', ha='center', fontsize=10)

# Objective
objective_width = 0.2
rim_thickness = 0.04
objective_height = 2
x_center = 5
bottom_y = 11.6
top_y = bottom_y + objective_height

ax.add_patch(Rectangle((x_center - objective_width / 2, bottom_y), objective_width, objective_height,
                       color='lightgray', ec='lightgray'))
ax.add_patch(Ellipse((x_center, top_y), objective_width, 0.2, color='dimgray', ec='dimgray'))

outer_r = (objective_width / 2) + rim_thickness / 2
inner_r = objective_width / 2

ax.add_patch(Wedge(center=(x_center, bottom_y), r=inner_r,
                   theta1=180, theta2=360, width=rim_thickness,
                   facecolor='lightgray', edgecolor='lightgray'))

ax.add_patch(Wedge(center=(x_center, bottom_y), r=inner_r,
                   theta1=180, theta2=360, width=inner_r,
                   facecolor='lightgray', edgecolor='none'))

# ax.text(x_center, bottom_y - 0.5, 'Objective', ha='center', fontsize=10)

# # Base
# base = Arc((5, 9.8), 10, 1.5, theta1=0, theta2=180, color='black')
# ax.add_patch(base)

plt.tight_layout()
plt.savefig("inverted_microscope_wider_sample_platform.png", dpi=300)
plt.show()
