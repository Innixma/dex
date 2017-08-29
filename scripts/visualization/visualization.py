# By Nick Erickson
# Contains functions for Visualization

import cv2
import imageio
import numpy as np
from vis.losses import ActivationMaximization
from vis.optimizer import Optimizer


# Normalizes image
def deprocess_image(img):
    # normalize tensor: center on 0., ensure std is 0.1
    img -= img.mean()
    img /= (img.std() + 1e-5)
    img *= 0.1
    img += 0.5
    img *= 255
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def visualize_saliency(model, layer_idx, filter_indices,
                       seed_img, text=None, overlay=False):
    heatmap = generate_heatmap(model, layer_idx, filter_indices, seed_img)
    heatmap_colored = heatmap_to_color(heatmap)

    overlayed_image = None
    text_image = None

    if overlay:
        tmp_img = seed_img[:,:,0]
        tmp_img = tmp_img.reshape(list(tmp_img.shape) + [1])
        new_img = np.repeat(tmp_img, 3, 2)
        new_img *= 255
        new_img = np.clip(new_img, 0, 255)
        new_img = new_img.astype('uint8')
        #new_img = seed_img

        overlayed_image = combine_images(new_img, 0.7, heatmap_colored, 0.7)
        if text:
            text_image = add_text_to_image(overlayed_image, text)

    return heatmap, heatmap_colored, overlayed_image, text_image

def generate_heatmap(model, layer_idx, filter_indices, seed_img):
    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), 1)
    ]
    opt = Optimizer(model.input, losses)
    grads = opt.minimize(max_iter=1, verbose=False, seed_img=seed_img)[1]

    # We are minimizing loss as opposed to maximizing output as with the paper.
    # So, negative gradients here mean that they reduce loss, maximizing class probability.
    grads *= -1
    grads = np.max(np.abs(grads), axis=3, keepdims=True)

    grads = deprocess_image(grads[0]).astype('float32') # Smoothen activation map
    grads = grads / np.max(grads) * 255

    # Convert to heatmap and zero out low probabilities for a cleaner output.
    heatmap = cv2.applyColorMap(cv2.GaussianBlur(grads, (3, 3), 0), cv2.COLORMAP_JET)

    heatmap = heatmap.reshape(list(heatmap.shape) + [1])
    heatmap[heatmap <= np.mean(heatmap)] = 0

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap) * 255

    return heatmap

def heatmap_to_color(heatmap, threshold=51):
    heatmap_colored = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)
    heatmap_colored[heatmap <= threshold] = 0
    return heatmap_colored

def combine_images(img1, weight1, img2, weight2):
    output = cv2.addWeighted(img1, weight1, img2, weight2, 0)
    return output

def add_text_to_image(img, text):
    new_img = np.copy(img)
    cv2.putText(new_img, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return new_img

def generate_saliceny_map(model, seed_imgs, show=True, text=None):
    """Generates a heatmap indicating the pixels that contributed the most towards
    maximizing the filter output. First, the class prediction is determined, then we generate heatmap
    to visualize that class.
    """

    layer_name = 'action' # The name of the layer we want to visualize
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    heatmaps = []
    heatmaps_c = []
    overlayed_images = []
    text_images = []
    i = -1
    maxIter = len(seed_imgs)
    for seed_img in seed_imgs:
        i += 1
        if i % 10 == 0:
            print('\r', 'Generating', '(', i, '/', maxIter, ')', end="")

        if text:
            curText = text[i]
        else:
            curText = None

        pred_class = np.argmax(model.predict(np.array([seed_img]))[0])
        heatmap, heatmap_colored, overlayed_image, text_image = visualize_saliency(model, layer_idx, [pred_class], seed_img, text=curText, overlay=True)

        heatmaps.append(heatmap)
        heatmaps_c.append(heatmap_colored)
        overlayed_images.append(overlayed_image)
        text_images.append(text_image)
        if show:
            cv2.imshow('Saliency', overlayed_image)
            cv2.waitKey(0)

    print('\r', 'Generating', '(', maxIter, '/', maxIter, ')')
    return heatmaps, heatmaps_c, overlayed_images, text_images
"""
args = hex_base_a3c_load
state_dim = [96, 96, 2]
action_dim = 3
memory_location = '../data/' + args.directory + '/'

agent = Agent(hex_base_a3c_load, state_dim, action_dim, modelFunc=models.CNN_a3c)
s, a, r, s_, t = loadMemory_direct(memory_location)

prevVal = 0
imminent_idx = []
for i in range(0, t.shape[0], 40):
    if prevVal == 1 and t[i] == 0:
        if i >= 40:
            imminent_idx.append(i-40)
    prevVal = t[i]
            
imminent_s = s[imminent_idx]

life_idx = []
for i in range(10600, 15800+200, 40):
    life_idx.append(i-40)

life_s = s[life_idx]
    
model = agent.brain.model
print('Model loaded.')

heatmaps, heatmaps_c, overlayed_images, text_images = generate_saliceny_map(model, life_s, show=False)
"""
def make_video(images, filename='opt_progress3.gif', fps=10, loop=1):
    writer = imageio.get_writer(filename, mode='I', loop=1, fps=10)
    c = 0
    numIter = len(images)
    for img in images:
        c += 1
        if c % 10 == 0:
            print(c, '/', numIter)
        writer.append_data(img)
    writer.close()
    print('Done!')

def rescale_images(images, scale=8):
    numIter = len(images)
    dim = np.array(list(images[0].shape[:2]))
    new_dims = tuple(dim*scale + [3])
    resized_images = []
    for i in range(numIter):
        resized = cv2.resize(images[i], new_dims, interpolation = cv2.INTER_AREA)
        resized_images.append(resized)
    return resized_images


def concat_3_videos(images1, images2, images3):
    numIter = len(images1)
    dim = list(images1[0].shape[:2])

    size = dim[0]

    new_size = size*2


    total_dim = tuple([new_size, new_size, 3])
    mid = int(size/2)

    print(total_dim)
    print(size)
    new_imgs = []
    for i in range(numIter):
        new_img = np.zeros(total_dim, dtype='uint8')
        new_img[:size, :size, :] = images1[i]
        new_img[:size, size:, :] = images2[i]
        new_img[size:, mid:mid+size, :] = images3[i]
        new_imgs.append(new_img)
    return new_imgs

def make_video_complex(images1, images2, filename='opt_progress2.gif', fps=10, loop=1):
    writer = imageio.get_writer(filename, mode='I', loop=1, fps=10)

    numIter = len(images1)
    dim = np.array(list(images1[0].shape[:2]))
    new_dims = tuple(dim*8 + [3])
    for i in range(numIter):
        if i % 10 == 0:
            print(i, '/', numIter)

        resized1 = cv2.resize(images1[i], new_dims, interpolation = cv2.INTER_AREA)
        resized2 = cv2.resize(images2[i], new_dims, interpolation = cv2.INTER_AREA)
        result = np.concatenate((resized1,resized2),0)
        writer.append_data(result)
    writer.close()
    print('Done!')
"""
real_images = []
for real in life_s:
    new_img = real[:,:,0]
    new_img = new_img.reshape(list(new_img.shape) + [1])
    new_img *= 255
    new_img = np.clip(new_img, 0, 255)
    new_img = new_img.astype('uint8')
    new_img = np.repeat(new_img, 3, 2)
    real_images.append(new_img)

print('rescaling')
real_images = rescale_images(real_images)
heatmaps_c = rescale_images(heatmaps_c)
overlayed_images = rescale_images(overlayed_images)

composite_images = concat_3_videos(real_images, heatmaps_c, overlayed_images)

make_video(composite_images)
"""
