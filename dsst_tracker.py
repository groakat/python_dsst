import numpy as np
import pylab as plt
import os
import time
import glob

import skimage.feature as skif
import skimage.transform as skit
import skimage.color as skic

from skimage.util import img_as_float

import pyhog


def repeat_to_third_dim(mat, repetitions=28):
    return np.tile(np.expand_dims(mat, 2), [repetitions])


def fhog(I, binSize=8, nOrients=9, clip=0.2, crop=0):
    return skif.hog(I,
                    orientations=nOrients,
                    pixels_per_cell=[binSize, binSize])


def get_feature_map(im_patch):
    out = np.zeros((im_patch.shape[0], im_patch.shape[1], 28))

    if len(im_patch.shape) == 2:
        out[:, :, 0] = im_patch - 0.5
        temp = pyhog.features_pedro(skic.gray2rgb(im_patch), 1)
        out[1:-1, 1:-1, 1:] = temp[:, :, :27]
    else:
        out[:, :, 0] = skic.rgb2gray(im_patch) - 0.5
        temp = pyhog.features_pedro(skic.gray2rgb(im_patch), 1)
        out[1:-1, 1:-1, 1:] = temp[:, :, :27]

    return out


def get_scale_sample(im, pos, base_target_sz, scaleFactors,
                     scale_window, scale_model_sz):
    nScales = len(scaleFactors)

    for s in range(nScales):
        patch_sz = np.floor(base_target_sz * scaleFactors[s])
        xs = np.floor(pos[1]) + np.arange(patch_sz[1]) - \
             np.floor(patch_sz[1] / 2)
        ys = np.floor(pos[0]) + np.arange(patch_sz[0]) - \
             np.floor(patch_sz[0] / 2)

        # # check for out-of-bounds coordinates, and set them to the values at
        # # the borders
        # xs[xs < 0] = 0
        # ys[ys < 0] = 0
        # xs[xs > im.shape[1] - 1] = im.shape[1] - 1
        # ys[ys > im.shape[0] - 1] = im.shape[0] - 1
        #
        # # extract image
        # im_patch = im[ys, xs, :]


        x_start = np.maximum(0, np.floor(pos[1]) - np.floor(patch_sz[1] / 2))
        y_start = np.maximum(0, np.floor(pos[0]) - np.floor(patch_sz[0] / 2))
        x_stop = np.minimum(im.shape[1] - 1,
                            np.floor(pos[1]) +
                            patch_sz[1] -
                            np.floor(patch_sz[1] / 2))
        y_stop = np.minimum(im.shape[0] - 1,
                            np.floor(pos[0]) +
                            patch_sz[0] -
                            np.floor(patch_sz[0] / 2))

        # extract image
        im_patch = im[x_start.astype(np.int32):x_stop.astype(np.int32),
                      y_start.astype(np.int32):y_stop.astype(np.int32)]

        # resize image to model size
        im_patch_resized = skit.resize(im_patch, scale_model_sz)

        temp_hog = pyhog.features_pedro(
            repeat_to_third_dim(im_patch_resized, 3), 4)

        temp = temp_hog[:, :, :32]

        if s == 0:
            out = np.zeros((temp.size, nScales))

        # window
        out[:, s] = temp.ravel() * scale_window[s]

    return out


def get_translation_sample(im, pos, model_sz, currentScaleFactor, cos_window):
    if np.isscalar(model_sz):
        model_sz = np.array([model_sz, model_sz])

    patch_sz = np.floor(model_sz * currentScaleFactor)

    # make sure the size is not too small
    if patch_sz[0] < 1:
        patch_sz[0] = 2

    if patch_sz[1] < 1:
        patch_sz[2] = 2

    # xs = np.floor(pos[1]) + np.arange(patch_sz[1]) - np.floor(patch_sz[1] / 2)
    # ys = np.floor(pos[0]) + np.arange(patch_sz[0]) - np.floor(patch_sz[0] / 2)
    #
    # # check for out-of-bounds coordinates, and set them to the values at
    # # the borders
    # xs[xs < 0] = 0
    # ys[ys < 0] = 0
    # xs[xs > im.shape[1] - 1] = im.shape[1] - 1
    # ys[ys > im.shape[0] - 1] = im.shape[0] - 1

    x_start = np.maximum(0, np.floor(pos[1]) - np.floor(patch_sz[1] / 2))
    y_start = np.maximum(0, np.floor(pos[0]) - np.floor(patch_sz[0] / 2))
    x_stop = np.minimum(im.shape[1] - 1,
                        np.floor(pos[1]) +
                        patch_sz[1] -
                        np.floor(patch_sz[1] / 2))
    y_stop = np.minimum(im.shape[0] - 1,
                        np.floor(pos[0]) +
                        patch_sz[0] -
                        np.floor(patch_sz[0] / 2))


    # extract image
    im_patch = im[y_start.astype(np.int32):y_stop.astype(np.int32),
                  x_start.astype(np.int32):x_stop.astype(np.int32)]

    # resize image to model size
    im_patch_r = skit.resize(im_patch, model_sz)

    # compute feature map
    out = get_feature_map(im_patch_r)
    # apply cosine window
    out = repeat_to_third_dim(cos_window) * out

    return out


def dsst(params):
    ground_truth = params['ground_truth']

    padding = params['padding']
    output_sigma_factor = params['output_sigma_factor']
    lmbda = params['lambda']
    learning_rate = params['learning_rate']
    nScales = params['number_of_scales']
    scale_step = params['scale_step']
    scale_sigma_factor = params['scale_sigma_factor']
    scale_model_max_area = params['scale_model_max_area']

    video_path = params['video_path']
    img_files = params['img_files']
    pos = np.floor(params['init_pos'])
    target_sz = np.floor(params['wsize'])

    visualization = params['visualization']

    num_frames = len(img_files)

    init_target_sz = target_sz

    #  target size att scale = 1
    base_target_sz = target_sz

    # window size, taking padding into account
    sz = np.floor(base_target_sz * (1 + padding))

    # desired translation filter output (gaussian shaped), bandwidth
    # proportional to target size
    output_sigma = np.sqrt(np.prod(base_target_sz)) * output_sigma_factor
    [rs, cs] = np.meshgrid(np.arange(1, sz[0] + 1) - np.floor(sz[0] / 2),
                           np.arange(1, sz[1] + 1) - np.floor(sz[1] / 2),
                           indexing='ij')

    y = np.exp(-0.5 * ((rs**2 + cs**2) / output_sigma**2))
    yf = np.fft.fft2(y, axes=(0, 1))

    # desired scale filter output (gaussian shaped), bandwidth proportional to
    # number of scales
    scale_sigma = nScales/np.sqrt(33) * scale_sigma_factor
    ss = np.arange(nScales) - np.ceil(nScales/2)
    ys = np.exp(-0.5 * (ss**2) / scale_sigma**2)
    ysf = np.fft.fft(ys)

    # store pre-computed translation filter cosine window
    # TODO: check shape. The transpose suggests sz should contain vectors
    cos_window = np.hanning(sz[0]).reshape(-1, 1) * np.hanning(sz[1])

    # store pre-computed scale filter cosine window
    if nScales % 2 == 0:
        scale_window = np.hanning(nScales + 1)
        scale_window = scale_window[1:]
    else:
        scale_window = np.hanning(nScales)

    ss = np.arange(nScales)
    scaleFactors = scale_step**(np.ceil(nScales / 2) - (ss + 1))

    # compute the resize dimensions used for feature extraction in the scale
    # estimation

    scale_model_factor = 1

    if np.prod(init_target_sz) > scale_model_max_area:
        scale_model_factor = np.sqrt(scale_model_max_area /
                                     np.prod(init_target_sz))

    scale_model_sz = np.floor(init_target_sz * scale_model_factor)
    currentScaleFactor = 1

    #to calculate precision
    positions = np.zeros((len(img_files), 4))

    #to calculate FPS
    tt = 0

    # find maximum and minimum scales
    im = plt.imread(os.path.join(video_path, img_files[0]))
    min_scale_factor = scale_step ** np.ceil(np.log(np.max(5 / sz)) /
                                             np.log(scale_step))
    max_scale_factor = scale_step ** \
                       np.floor(np.log(np.min(np.array([im.shape[0],
                                                        im.shape[1]]) /
                                              base_target_sz)) /
                                np.log(scale_step))

    if visualization == 1:
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

    # TODO: TESTED UP TO HERE
    for frame in range(0, num_frames):
        # load image
        im = img_as_float(
                plt.imread(
                    os.path.join(video_path, img_files[frame])))
        if len(im.shape) == 3:
            im = skic.rgb2gray(im)

        t1 = time.time()

        if frame > 0:
            # extract the test sample feature map for the translation filter
            xt = get_translation_sample(im,
                                        pos,
                                        sz,
                                        currentScaleFactor,
                                        cos_window)

            # calculate the correlation response of the translation filter
            xtf = np.fft.fft2(xt, axes=(0, 1))
            response = np.real(np.fft.ifft2(np.sum(hf_num * xtf, 2) /
                                            (hf_den + lmbda), axes=(0, 1)))

            if visualization == 1:

                if frame == 1:
                    imshow_obj_2 = ax2.imshow(response, cmap=plt.cm.viridis)
                    imshow_obj_3 = ax3.imshow(np.real(np.fft.ifft2(hf_num, axes=(0, 1))[..., 0]),
                               cmap=plt.cm.viridis)
                    imshow_obj_4 = ax4.imshow(np.real(np.fft.ifft2(xtf, axes=(0, 1))[..., 0]),
                               cmap=plt.cm.viridis)
                else:
                    imshow_obj_2.set_data(response)
                    imshow_obj_3.set_data(np.real(np.fft.ifft2(hf_num, axes=(0, 1))[..., 0]))
                    imshow_obj_4.set_data(np.real(np.fft.ifft2(xtf, axes=(0, 1))[..., 0]))

                    fig.canvas.draw()

            # find the maximum translation response
            [row, col] = np.argwhere(response == np.max(response.ravel()))[0]

            # update the position
            pos += np.round((-sz/2 + np.array([row, col]) + 1) * currentScaleFactor)

            print(frame, pos)


            # extract the test sample feature map for the scale filter
            xs = get_scale_sample(im,
                                  pos,
                                  base_target_sz,
                                  currentScaleFactor * scaleFactors,
                                  scale_window,
                                  scale_model_sz)

            # calculate the correlation response of the scale filter
            xsf = np.fft.fft(xs, axis=1)
            scale_response = np.real(np.fft.ifft(np.sum(sf_num * xsf, axis=0) /
                                                 (sf_den + lmbda)))

            # find the maximum scale response
            # TODO find out why the -1 hack produces better scale estimates
            recovered_scale = np.argwhere(
                scale_response == np.max(scale_response.ravel()))[0] - 1

            # recovered_scale = 16
            print(recovered_scale)
            # update the scale
            currentScaleFactor = currentScaleFactor * \
                                 scaleFactors[recovered_scale]

            if currentScaleFactor < min_scale_factor:
                currentScaleFactor = min_scale_factor
            elif currentScaleFactor > max_scale_factor:
                currentScaleFactor = max_scale_factor

        # extract the training sample feature map for the translation filter
        xl = get_translation_sample(im, pos, sz, currentScaleFactor, cos_window)

        # calculate the translation filter update
        xlf = np.fft.fft2(xl, axes=(0, 1))
        new_hf_num = repeat_to_third_dim(yf) * np.conj(xlf)
        new_hf_den = np.sum(xlf * np.conj(xlf), axis=2)

        # extract the training sample feature map for the scale filter
        xs = get_scale_sample(im,
                              pos,
                              base_target_sz,
                              currentScaleFactor * scaleFactors,
                              scale_window,
                              scale_model_sz)

        # calculate the scale filter update
        xsf = np.fft.fft(xs, axis=1)
        new_sf_num = ysf * np.conj(xsf)
        new_sf_den = np.sum(xsf * np.conj(xsf), axis=0)

        if frame == 0:
            # first frame, train with a single image
            hf_den = new_hf_den
            hf_num = new_hf_num

            sf_den = new_sf_den
            sf_num = new_sf_num
        else:
            # subsequent frames, update the model
            hf_den = (1 - learning_rate) * hf_den + learning_rate * new_hf_den
            hf_num = (1 - learning_rate) * hf_num + learning_rate * new_hf_num
            sf_den = (1 - learning_rate) * sf_den + learning_rate * new_sf_den
            sf_num = (1 - learning_rate) * sf_num + learning_rate * new_sf_num

        # calculate the new target size
        target_sz = np.floor(base_target_sz * currentScaleFactor)

        # save position
        positions[frame, :] = np.concatenate([pos, target_sz])

        tt += time.time() - t1

        if visualization == 1:
        #     if frame == 0:
        #         fig = plt.figure()
        #         ax = fig.add_subplot(111)
        #         plt.ylim([300, 0])
        #         plt.xlim([0, 300])
        #
            if frame == 0:
                imshow_obj_1 = ax1.imshow(im, cmap=plt.cm.gray)
                ax1.plot(pos[1], pos[0], c='blue', marker='.')
                ax1.plot(ground_truth[frame, 0], ground_truth[frame, 1],
                        c='green', marker='.')
            else:
                ax1.plot(pos[1], pos[0], c='blue', marker='.')
                ax1.plot(ground_truth[frame, 0], ground_truth[frame, 1],
                        c='green', marker='.')
                imshow_obj_1.set_data(im)
                plt.draw()
        #     plt.gcf().canvas.draw()
        #     fig.clear()



    print("fps: {}".format(num_frames/tt))

    return positions


def main():
    # video_path = "sequences/dog1"
    target_sz = np.array([36, 51])
    video_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'sequences/dog1/imgs/')

    import pandas as pd
    ground_truth = np.asarray(pd.read_csv('sequences/dog1/dog1_gt.txt',
                                          header=None))

    params = {"padding": 1.0,
              "output_sigma_factor": 1 / 16.0,
              "scale_sigma_factor": 1 / 4.0,
              "lambda": 1e-2,
              "learning_rate": 0.025,
              "number_of_scales": 33,
              "scale_step": 1.02,
              "scale_model_max_area": 512,
              "visualization": 1,
              "init_pos": np.array([112, 139]) + target_sz / 2,
              "wsize": np.floor(target_sz),
              "img_files": glob.glob(os.path.join(video_path, '*.jpg')),
              "video_path": video_path,
              'ground_truth':ground_truth
              }

    return dsst(params)
