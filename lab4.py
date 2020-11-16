""" CS4243 Lab 4: Tracking
Please read accompanying Jupyter notebook (lab4.ipynb) and PDF (lab4.pdf) for instructions.
"""
import cv2
import numpy as np
import random
import math

from time import time
from skimage import filters


# Part 1 

def meanShift(dst, track_window, max_iter=100,stop_thresh=1):
    """Use mean shift algorithm to find an object on a back projection image.

    Args:
        dst (np.ndarray)            : Back projection of the object histogram of shape (H, W).
        track_window (tuple)        : Initial search window. (x,y,w,h)
        max_iter (int)              : Max iteration for mean shift.
        stop_thresh(float)          : Threshold for convergence.
    
    Returns:
        track_window (tuple)        : Final tracking result. (x,y,w,h)


    """

    completed_iterations = 0
    
    """ YOUR CODE STARTS HERE """
    shift = float('inf')
    
    center_x = math.floor(track_window[2]/2)
    center_y = math.floor(track_window[3]/2)
    
    max_y,max_x = dst.shape
    
    while completed_iterations < max_iter and shift > stop_thresh:
        completed_iterations += 1
        
        x,y,w,h = track_window
        
        window = dst[y:y+h,x:x+w]
        
        start_x = -center_x
        start_y = -center_y
        end_x = center_x
        end_y = center_y
        
        if (x < 0):
            start_x = start_x-x
        if (y < 0):
            start_y = start_y-y
        if (x + w > max_x):
            end_x = end_x - (x + w - max_x)
        if (y + h > max_y):
            end_y = end_y - (y + h - max_y)
        
        x_coordinate = np.arange(start_x,end_x+1)
        y_coordinate = np.arange(start_y,end_y+1)
        
        sum_x = window.sum(axis=0)
        sum_y = window.sum(axis=1)
        
        xBarA = np.dot(x_coordinate,sum_x)
        yBarA = np.dot(y_coordinate,sum_y)
        
        shift_x = round(xBarA/np.sum(sum_x))
        shift_y = round(yBarA/np.sum(sum_y))

        #update the point
        track_window = (track_window[0] + shift_x,track_window[1] + shift_y,w,h)
        
        shift = shift_x**2+shift_y**2
    """ YOUR CODE ENDS HERE """
    
    return track_window
    
    
    
        

def IoU(bbox1, bbox2):
    """ Compute IoU of two bounding boxes.

    Args:
        bbox1 (tuple)               : First bounding box position (x, y, w, h) where (x, y) is the top left corner of
                                      the bounding box, and (w, h) are width and height of the box.
        bbox2 (tuple)               : Second bounding box position (x, y, w, h) where (x, y) is the top left corner of
                                      the bounding box, and (w, h) are width and height of the box.
    Returns:
        score (float)               : computed IoU score.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    score = 0

    """ YOUR CODE STARTS HERE """
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1+w1, x2+w2)
    yB = min(y1+h1, y2+h2)

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxA = (w1 + 1) * (h1 + 1)
    boxB = (w2 + 1) * (h2 + 1)

    score = interArea / float(boxA + boxB - interArea)
    """ YOUR CODE ENDS HERE """

    return score


# Part 2:
def lucas_kanade(img1, img2, keypoints, window_size=9):
    """ Estimate flow vector at each keypoint using Lucas-Kanade method.

    Args:
        img1 (np.ndarray)           : Grayscale image of the current frame. 
                                      Flow vectors are computed with respect to this frame.
        img2 (np.ndarray)           : Grayscale image of the next frame.
        keypoints (np.ndarray)      : Coordinates of keypoints to track of shape (N, 2).
        window_size (int)           : Window size to determine the neighborhood of each keypoint.
                                      A window is centered around the current keypoint location.
                                      You may assume that window_size is always an odd number.
    Returns:
        flow_vectors (np.ndarray)   : Estimated flow vectors for keypoints. flow_vectors[i] is
                                      the flow vector for keypoint[i]. Array of shape (N, 2).

    Hints:
        - You may use np.linalg.inv to compute inverse matrix.
    """
    assert window_size % 2 == 1, "window_size must be an odd number"
    
    flow_vectors = []
    w = window_size // 2

    # Compute partial derivatives
    Iy, Ix = np.gradient(img1)
    It = img2 - img1

    # For each [y, x] in keypoints, estimate flow vector [vy, vx]
    # using Lucas-Kanade method and append it to flow_vectors.
    for y, x in keypoints:
        # Keypoints can be located between integer pixels (subpixel locations).
        # For simplicity, we round the keypoint coordinates to nearest integer.
        # In order to achieve more accurate results, image brightness at subpixel
        # locations can be computed using bilinear interpolation.
        y, x = int(round(y)), int(round(x))

        """ YOUR CODE STARTS HERE """
        window_x = np.zeros((3,3))
        window_y = np.zeros((3,3))
        
        start_y = y - int(window_size/2)
        start_x = x - int(window_size/2)
        end_y = y + int(window_size/2)
        end_x = x + int(window_size/2)
        
        A = np.zeros((window_size*window_size, 2))
        b = np.zeros(window_size*window_size)
        index = 0
        for i in range(start_y, end_y + 1, 1):
            for j in range(start_x, end_x + 1, 1):
                A[index][0] = Iy[i][j]
                A[index][1] = Ix[i][j]
                b[index] = It[i][j]
                index = index + 1
        
        At = np.transpose(A)
        At_A = np.matmul(At, A)
        x_hat = np.matmul(np.linalg.inv(At_A),np.matmul(At, b))
        flow_vectors.append(x_hat)
        

        """ YOUR CODE ENDS HERE """

    flow_vectors = np.array(flow_vectors)

    return flow_vectors


def compute_error(patch1, patch2):
    """ Compute MSE between patch1 and patch2
        - Normalize patch1 and patch2
        - Compute mean square error between patch1 and patch2

    Args:
        patch1 (np.ndarray)         : Grayscale image patch1 of shape (patch_size, patch_size)
        patch2 (np.ndarray)         : Grayscale image patch2 of shape (patch_size, patch_size)
    Returns:
        error (float)               : Number representing mismatch between patch1 and patch2.
    """
    assert patch1.shape == patch2.shape, 'Differnt patch shapes'
    error = 0

    """ YOUR CODE STARTS HERE """
    sd1 = np.std(patch1, dtype=np.float64)
    sd2 = np.std(patch2, dtype=np.float64)
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)
    
    norm_patch1 = (patch1-mean1) / sd1
    norm_patch2 = (patch2-mean2) / sd2
    
    error = np.mean(np.square(norm_patch2-norm_patch1))
    

    """ YOUR CODE ENDS HERE """

    return error



def iterative_lucas_kanade(img1, img2, keypoints,
                           window_size=9,
                           num_iters=5,
                           g=None):
    """ Estimate flow vector at each keypoint using iterative Lucas-Kanade method.

    Args:
        img1 (np.ndarray)           : Grayscale image of the current frame. 
                                      Flow vectors are computed with respect to this frame.
        img2 (np.ndarray)           : Grayscale image of the next frame.
        keypoints (np.ndarray)      : Coordinates of keypoints to track of shape (N, 2).
        window_size (int)           : Window size to determine the neighborhood of each keypoint.
                                      A window is centered around the current keypoint location.
                                      You may assume that window_size is always an odd number.

        num_iters (int)             : Number of iterations to update flow vector.
        g (np.ndarray)              : Flow vector guessed from previous pyramid level.
                                      Array of shape (N, 2).
    Returns:
        flow_vectors (np.ndarray)   : Estimated flow vectors for keypoints. flow_vectors[i] is
                                      the flow vector for keypoint[i]. Array of shape (N, 2).
    """
    assert window_size % 2 == 1, "window_size must be an odd number"

    # Initialize g as zero vector if not provided
    if g is None:
        g = np.zeros(keypoints.shape)

    flow_vectors = []
    w = window_size // 2
    
   
    # Compute spatial gradients
    Iy, Ix = np.gradient(img1)

    for y, x, gy, gx in np.hstack((keypoints, g)):
        v = np.zeros(2) # Initialize flow vector as zero vector
        y1 = int(round(y)); x1 = int(round(x))
        
        """ YOUR CODE STARTS HERE """
        start_y = y1 - int(window_size/2)
        start_x = x1 - int(window_size/2)
        end_y = y1 + int(window_size/2)
        end_x = x1 + int(window_size/2)
        
        A = np.zeros((window_size*window_size, 2))
        index = 0
        for i in range(start_y, end_y + 1, 1):
            for j in range(start_x, end_x + 1, 1):
                A[index][0] = Iy[i][j]
                A[index][1] = Ix[i][j]
                index += 1

        At = np.transpose(A)
        At_A = np.matmul(At, A) 
        G = np.array([[At_A[1,1], At_A[0,1]],[At_A[1,0], At_A[0,0]]])
        window_Ix = A[:, 1].reshape((window_size, window_size))
        window_Iy = A[:, 0].reshape((window_size, window_size))

        for k in range (num_iters):
            I_present = img1[(y1 - w) : (y1 + w + 1), (x1 - w) : (x1 + w + 1)]
            J_next = img2[(y1 - w + int(gy) + int(v[1]) ) : (y1 + w + 1 + int(gy) + int(v[1])), (x1 - w + int(gx) + int(v[0])) : (x1 + w + 1 + int(gx) + int(v[0]))]
            temp_diff = I_present - J_next 
            b_k = np.array([np.sum(temp_diff* window_Ix),np.sum(temp_diff * window_Iy)])
            v += np.matmul(np.linalg.inv(G), b_k)
    

        """ YOUR CODE ENDS HERE """

        vx, vy = v
        flow_vectors.append([vy, vx])
        
    return np.array(flow_vectors)
        

def pyramid_lucas_kanade(img1, img2, keypoints,
                         window_size=9, num_iters=5,
                         level=2, scale=2):

    """ Pyramidal Lucas Kanade method

    Args:
        img1 (np.ndarray)           : Grayscale image of the current frame. 
                                      Flow vectors are computed with respect to this frame.
        img2 (np.ndarray)           : Grayscale image of the next frame.
        keypoints (np.ndarray)      : Coordinates of keypoints to track of shape (N, 2).
        window_size (int)           : Window size to determine the neighborhood of each keypoint.
                                      A window is centered around the current keypoint location.
                                      You may assume that window_size is always an odd number.

        num_iters (int)             : Number of iterations to run iterative LK method
        level (int)                 : Max level in image pyramid. Original image is at level 0 of
                                      the pyramid.
        scale (float)               : Scaling factor of image pyramid.

    Returns:
        d - final flow vectors
    """

    # Build image pyramids of img1 and img2
    pyramid1 = tuple(pyramid_gaussian(img1, max_layer=level, downscale=scale))
    pyramid2 = tuple(pyramid_gaussian(img2, max_layer=level, downscale=scale))

    # Initialize pyramidal guess
    g = np.zeros(keypoints.shape)

    """ YOUR CODE STARTS HERE """
#     for L in range(level, -1, -1):
#         keypoints_L = keypoints / (scale ** L)
#         d = iterative_lucas_kanade(pyramid1[L], pyramid2[L], keypoints_L, g=g)
#         if L != 0:
#             g = scale * (g + d)

    L_m = level
    kp_on_L = keypoints * scale
    
    for L in range(level, -1, -1):
        # Compute location of p on I^L
        kp_on_L = kp_on_L / scale**L
        
        # Compute optical flow vector at level L
        d = iterative_lucas_kanade(pyramid1[L], pyramid2[L], kp_on_L, window_size, num_iters, g=g)
        
        # Guess for next level
        g = scale*(g + d)

    """ YOUR CODE ENDS HERE """

    d = g + d
    return d























"""Helper functions: You should not have to touch the following functions.
"""
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

from skimage import filters, img_as_float
from skimage.io import imread
from skimage.transform import pyramid_gaussian

def load_frames_rgb(imgs_dir):

    frames = [cv2.cvtColor(cv2.imread(os.path.join(imgs_dir, frame)), cv2.COLOR_BGR2RGB) \
              for frame in sorted(os.listdir(imgs_dir))]
    return frames

def load_frames_as_float_gray(imgs_dir):
    frames = [img_as_float(imread(os.path.join(imgs_dir, frame), 
                                               as_gray=True)) \
              for frame in sorted(os.listdir(imgs_dir))]
    return frames

def load_bboxes(gt_path):
    bboxes = []
    with open(gt_path) as f:
        for line in f:
          
            x, y, w, h = line.split(',')
            #x, y, w, h = line.split()
            bboxes.append((int(x), int(y), int(w), int(h)))
    return bboxes

def animated_frames(frames, figsize=(10,8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0])

    def animate(i):
        im.set_array(frames[i])
        return [im,]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=60, blit=True)

    return ani

def animated_bbox(frames, bboxes, figsize=(10,8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0])
    x, y, w, h = bboxes[0]
    bbox = ax.add_patch(Rectangle((x,y),w,h, linewidth=3,
                                  edgecolor='r', facecolor='none'))

    def animate(i):
        im.set_array(frames[i])
        bbox.set_bounds(*bboxes[i])
        return [im, bbox,]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=60, blit=True)

    return ani

def animated_scatter(frames, trajs, figsize=(10,8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0])
    scat = ax.scatter(trajs[0][:,1], trajs[0][:,0],
                      facecolors='none', edgecolors='r')

    def animate(i):
        im.set_array(frames[i])
        if len(trajs[i]) > 0:
            scat.set_offsets(trajs[i][:,[1,0]])
        else: # If no trajs to draw
            scat.set_offsets([]) # clear the scatter plot

        return [im, scat,]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=60, blit=True)

    return ani

def track_features(frames, keypoints,
                   error_thresh=1.5,
                   optflow_fn=pyramid_lucas_kanade,
                   exclude_border=5,
                   **kwargs):

    """ Track keypoints over multiple frames

    Args:
        frames - List of grayscale images with the same shape.
        keypoints - Keypoints in frames[0] to start tracking. Numpy array of
            shape (N, 2).
        error_thresh - Threshold to determine lost tracks.
        optflow_fn(img1, img2, keypoints, **kwargs) - Optical flow function.
        kwargs - keyword arguments for optflow_fn.

    Returns:
        trajs - A list containing tracked keypoints in each frame. trajs[i]
            is a numpy array of keypoints in frames[i]. The shape of trajs[i]
            is (Ni, 2), where Ni is number of tracked points in frames[i].
    """

    kp_curr = keypoints
    trajs = [kp_curr]
    patch_size = 3 # Take 3x3 patches to compute error
    w = patch_size // 2 # patch_size//2 around a pixel

    for i in range(len(frames) - 1):
        I = frames[i]
        J = frames[i+1]
        flow_vectors = optflow_fn(I, J, kp_curr, **kwargs)
        kp_next = kp_curr + flow_vectors

        new_keypoints = []
        for yi, xi, yj, xj in np.hstack((kp_curr, kp_next)):
            # Declare a keypoint to be 'lost' IF:
            # 1. the keypoint falls outside the image J
            # 2. the error between points in I and J is larger than threshold

            yi = int(round(yi)); xi = int(round(xi))
            yj = int(round(yj)); xj = int(round(xj))
            # Point falls outside the image
            if yj > J.shape[0]-exclude_border-1 or yj < exclude_border or\
               xj > J.shape[1]-exclude_border-1 or xj < exclude_border:
                continue

            # Compute error between patches in image I and J
            patchI = I[yi-w:yi+w+1, xi-w:xi+w+1]
            patchJ = J[yj-w:yj+w+1, xj-w:xj+w+1]
            error = compute_error(patchI, patchJ)
            if error > error_thresh:
                continue

            new_keypoints.append([yj, xj])

        kp_curr = np.array(new_keypoints)
        trajs.append(kp_curr)

    return trajs