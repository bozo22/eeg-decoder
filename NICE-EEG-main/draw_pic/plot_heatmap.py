import mne
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties 
# font = FontProperties(fname=r"simsun.ttc", size=8) 

MODEL_NAME = 'SuperNICE'

results = []
for sub in range(1, 11): 
    tmp = np.load(os.path.join('NICE-EEG-main/results', MODEL_NAME, 'similarity', f'sub{sub}_sim.npy'))
    results.append(tmp)
results = np.array(results)
results = np.mean(results, axis=0)
results = (results - np.min(results)) / (np.max(results) - np.min(results))
# Drop results below certain similarity threshold
threshold = 0.3
results[results < threshold] = 0


# Split test set into large categories
animal_idxs = [2, 12, 24, 33, 34, 39, 46, 58, 60, 61, 63, 65, 69, 70, 72, 76, 86, 87, 88, 89, 97, 106, 110, 111, 117, 127, 129, 133, 136, 137, 142, 144, 150, 152, 161, 183, 190]
food_idxs = [5, 7, 11, 15, 18, 22, 27, 29, 32, 38, 47, 48, 50, 51, 53, 54, 55, 57, 64, 71, 73, 81, 82, 91, 98, 99, 101, 109, 113, 120, 122, 123, 124, 125, 131, 132, 135, 140, 141, 143, 147, 148, 157, 158, 159, 174, 184, 195, 196, 199]
vehicle_idxs = [1, 14, 17, 25, 31, 59, 75, 84, 85, 100, 115, 154, 160, 165, 172, 175, 191, 197]
tool_idxs = [3, 4, 6, 8, 9, 10, 16, 20, 30, 41, 42, 44, 49, 62, 67, 78, 80, 92, 103, 105, 114, 116, 119, 134, 139, 145, 164, 168, 170, 171, 173, 180, 185, 186, 192, 193, 200]
clothing_idxs = [19, 37, 43, 45, 52, 68, 83, 94, 96, 104, 118, 128, 138, 146, 155, 169, 176, 177, 182, 187, 189]
others_idxs = [13, 21, 23, 26, 28, 35, 36, 40, 56, 66, 74, 77, 79, 90, 93, 95, 102, 107, 108, 112, 121, 126, 130, 149, 151, 153, 156, 162, 163, 166, 167, 178, 179, 181, 188, 194, 198]
idxs = animal_idxs + food_idxs + vehicle_idxs + tool_idxs + clothing_idxs + others_idxs
idxs = [x - 1 for x in idxs]
results = results[idxs][:, idxs]

plt.figure(figsize=(14, 12))
plt.imshow(results, cmap='Blues')  # Available colormaps include:
# Sequential: viridis, plasma, inferno, magma, cividis
# Sequential2: Greys, Purples, Blues, Greens, Oranges, Reds, YlOrBr, YlOrRd, OrRd, PuRd, RdPu, BuPu, GnBu, PuBu, YlGnBu, PuBuGn, BuGn, YlGn
# Diverging: PiYG, PRGn, BrBG, PuOr, RdGy, RdBu, RdYlBu, RdYlGn, Spectral, coolwarm, bwr, seismic
# Qualitative: Pastel1, Pastel2, Paired, Accent, Dark2, Set1, Set2, Set3, tab10, tab20, tab20b, tab20c
# Misc: flag, prism, ocean, gist_earth, terrain, gist_stern, gnuplot, gnuplot2, CMRmap, cubehelix, brg, gist_rainbow, rainbow, jet, turbo, nipy_spectral

last_animal_idx = len(animal_idxs)
last_food_idx = last_animal_idx + len(food_idxs)
last_vehicle_idx = last_food_idx + len(vehicle_idxs)
last_tool_idx = last_vehicle_idx + len(tool_idxs)
last_clothing_idx = last_tool_idx + len(clothing_idxs)
last_others_idx = last_clothing_idx + len(others_idxs)


plt.plot([0, last_animal_idx], [last_animal_idx, last_animal_idx], color='black', alpha=0.3)
plt.plot([last_animal_idx, last_animal_idx], [0, last_animal_idx], color='black', alpha=0.3)
plt.plot([0, 0], [0, last_animal_idx], color='black', alpha=0.3)
plt.plot([0, len(animal_idxs)], [0, 0], color='black', alpha=0.3)

plt.plot([last_animal_idx, last_food_idx], [last_food_idx, last_food_idx], color='black', alpha=0.3)
plt.plot([last_food_idx, last_food_idx], [last_animal_idx, last_food_idx], color='black', alpha=0.3)
plt.plot([last_animal_idx, last_animal_idx], [last_animal_idx, last_food_idx], color='black', alpha=0.3)
plt.plot([last_animal_idx, last_food_idx], [last_animal_idx, last_animal_idx], color='black', alpha=0.3)


plt.plot([last_food_idx, last_vehicle_idx], [last_vehicle_idx, last_vehicle_idx], color='black', alpha=0.3)
plt.plot([last_vehicle_idx, last_vehicle_idx], [last_food_idx, last_vehicle_idx], color='black', alpha=0.3)
plt.plot([last_food_idx, last_food_idx], [last_food_idx, last_vehicle_idx], color='black', alpha=0.3)
plt.plot([last_food_idx, last_vehicle_idx], [last_food_idx, last_food_idx], color='black', alpha=0.3)

plt.plot([last_vehicle_idx, last_tool_idx], [last_tool_idx, last_tool_idx], color='black', alpha=0.3)
plt.plot([last_tool_idx, last_tool_idx], [last_vehicle_idx, last_tool_idx], color='black', alpha=0.3)
plt.plot([last_vehicle_idx, last_vehicle_idx], [last_vehicle_idx, last_tool_idx], color='black', alpha=0.3)
plt.plot([last_vehicle_idx, last_tool_idx], [last_vehicle_idx, last_vehicle_idx], color='black', alpha=0.3)

plt.plot([last_tool_idx, last_clothing_idx], [last_clothing_idx, last_clothing_idx], color='black', alpha=0.3)
plt.plot([last_clothing_idx, last_clothing_idx], [last_tool_idx, last_clothing_idx], color='black', alpha=0.3)
plt.plot([last_tool_idx, last_tool_idx], [last_tool_idx, last_clothing_idx], color='black', alpha=0.3)
plt.plot([last_tool_idx, last_clothing_idx], [last_tool_idx, last_tool_idx], color='black', alpha=0.3)

plt.plot([last_clothing_idx, 200], [200, 200], color='black', alpha=0.3)
plt.plot([200, 200], [last_clothing_idx, 200], color='black', alpha=0.3)
plt.plot([last_clothing_idx, last_clothing_idx], [last_clothing_idx, 200], color='black', alpha=0.3)
plt.plot([last_clothing_idx, 200], [last_clothing_idx, last_clothing_idx], color='black', alpha=0.3)

plt.colorbar(shrink=0.8)

x_ticks = [last_animal_idx // 2, last_animal_idx + len(food_idxs) // 2, last_food_idx + len(vehicle_idxs) // 2, last_vehicle_idx + len(tool_idxs) // 2, last_tool_idx + len(clothing_idxs) // 2, last_clothing_idx + len(others_idxs) // 2]
x_ticklabels = ['animal', 'food', 'vehicle', 'tool', 'clothing', 'others']


plt.xlim(0, 200)
plt.ylim(200, 0)
plt.xticks(x_ticks, x_ticklabels, fontsize=18)
plt.yticks(x_ticks, x_ticklabels, fontsize=18, rotation=90)

plt.xlabel('Image features', fontsize=20, fontweight='bold')
plt.ylabel('EEG features', fontsize=20, fontweight='bold')


plt.title('Similarity', size=20, fontweight='bold')
plt.tight_layout()
plt.savefig('NICE-EEG-main/draw_pic/heatmap1.png', dpi=300)
# plt.show()

print('the end')
