import numpy as np
import matplotlib.pyplot as plt
import sys
img_name = sys.argv[1]


wh = ''.join([x for x in img_name if x.isdigit()]) # 圖片邊長
wh = int(wh)
if (len(sys.argv) >= 3):
    wh -= int(sys.argv[2])
img = np.fromfile(img_name, dtype='uint8')
img = img.reshape([wh, wh])
plt.imshow(img,cmap="gray")
print(img.shape)
save_img_name = img_name.split('.bin')[0] + '.png'
plt.savefig(save_img_name)
# plt.show()