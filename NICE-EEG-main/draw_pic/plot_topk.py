import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os 

MODEL_NAME = 'SuperNICE'
sub_idx = 1
y_true = np.arange(200)
# y_true = y_true.view(-1, 1)
y_pred = np.load(os.path.join('NICE-EEG-main/results', MODEL_NAME, 'similarity', f'sub{sub_idx}_sim.npy'))
y_pred = np.argsort(y_pred, axis=1)[:, -5:][:, ::-1]

# Find indices where top5 predictions contain the true label
match = np.array([y_true[i] in y_pred[i] for i in range(len(y_true))])
match_place = np.where(match)[0]

things_eeg_test_images_path = 'Things-EEG2/Image_set/test_images'
things_list = [x for x in os.listdir(things_eeg_test_images_path) if x.startswith('00')]
things_list.sort()
print(f"Things list is: {things_list}")

fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(10, 8), gridspec_kw={'wspace': 0.1, 'hspace': 0.2})
place = [6, 9, 31, 43, 61]

for i in range(5):
    # plt.subplot(5, 6, 6*i+1)
    tmp_idx = y_true[match_place[place[i]]]
    tmp_img_file = os.listdir(os.path.join(things_eeg_test_images_path, things_list[tmp_idx]))[0]
    tmp_img = Image.open(os.path.join(things_eeg_test_images_path, things_list[tmp_idx], tmp_img_file))
    # plt.imshow(tmp_img)
    # # plt.axis('off')
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Ground Truth')
    axes[i, 0].imshow(tmp_img)
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])
    axes[i, 0].set_xlabel(things_list[tmp_idx][6:], fontsize=12, labelpad=2)
    if i == 0:
        axes[i, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')

    for j in range(5):
        # plt.subplot(5, 6, 6*i+j+2)
        tmp_idx = y_pred[match_place[place[i]]][j]
        tmp_img_file = os.listdir(os.path.join(things_eeg_test_images_path, things_list[tmp_idx]))[0]
        tmp_img = Image.open(os.path.join(things_eeg_test_images_path, things_list[tmp_idx], tmp_img_file))
        axes[i, j+1].imshow(tmp_img)
        if i == 0:
            axes[i, j+1].set_title('Top %d' % (j+1), fontsize=14)
        axes[i, j+1].set_xticks([])
        axes[i, j+1].set_yticks([])
        axes[i, j+1].spines['top'].set_visible(False)
        axes[i, j+1].spines['right'].set_visible(False)
        axes[i, j+1].spines['bottom'].set_visible(False)
        axes[i, j+1].spines['left'].set_visible(False)
        axes[i, j+1].set_xlabel(things_list[tmp_idx][6:], fontsize=12, labelpad=2)

plt.savefig('NICE-EEG-main/draw_pic/topk.png', dpi=300)
# plt.show()


