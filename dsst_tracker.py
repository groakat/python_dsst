import numpy as np
import pylab as plt
import os
import time

import skimage.feature as skif
import skimage.transform as skit
import skimage.color as skic


def fhog(I, binSize=8, nOrients=9, clip=0.2, crop=0):
    return skif.hog(I,
                    orientations=nOrients,
                    pixels_per_cell=[binSize, binSize])


def get_feature_map(im_patch):
    out = np.zeros((im_patch.shape[0], im_patch.shape[1], 28))

    if len(im_patch.shape) == 2:
        out[:, :, 0] = im_patch / 255 - 0.5
        temp = fhog(im_patch, 1)
        out[:, :, 1:] = temp[:, :, :27]
    else:
        out[:, :, 0] = skic.rgb2gray(im_patch) / 255 - 0.5
        temp = fhog(im_patch, 1)
        out[:, :, 1:] = temp[:, :, :27]

    return out


def get_scale_sample(im, pos, base_target_sz, scaleFactors,
                     scale_window, scale_model_sz):
    nScales = len(scaleFactors)

    for s in nScales:
        patch_sz = np.floor(base_target_sz * scaleFactors[s])
        xs = np.floor(pos[1]) + np.arange(patch_sz[1]) - \
             np.floor(patch_sz[1] / 2)
        ys = np.floor(pos[0]) + np.arange(patch_sz[0]) - \
             np.floor(patch_sz[0] / 2)

        # check for out-of-bounds coordinates, and set them to the values at
        # the borders
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs > im.shape[1] - 1] = im.shape[1] - 1
        ys[ys > im.shape[0] - 1] = im.shape[0] - 1

        # extract image
        im_patch = im[ys, xs, :]

        # resize image to model size
        im_patch_resized = skit.resize(im_patch, scale_model_sz)

        temp_hog = fhog(im_patch_resized, 4)
        temp = temp_hog[:, :, :30]

        if s == 1:
            out = np.zeros((len(temp), nScales))

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

    xs = np.floor(pos[1]) + np.arange(patch_sz[1]) - np.floor(patch_sz[1] / 2)
    ys = np.floor(pos[0]) + np.arange(patch_sz[0]) - np.floor(patch_sz[0] / 2)

    # check for out-of-bounds coordinates, and set them to the values at
    # the borders
    xs[xs < 0] = 0
    ys[ys < 0] = 0
    xs[xs > im.shape[1] - 1] = im.shape[1] - 1
    ys[ys > im.shape[0] - 1] = im.shape[0] - 1

    # extract image
    im_patch = im[ys, xs, :]

    # resize image to model size
    im_patch = skit.resize(im_patch, model_sz)

    # compute feature map
    out = get_feature_map(im_patch)

    # apply cosine window
    out = cos_window * out

    return out


def dsst(params):
    padding = params['padding']
    output_sigma_factor = params['output_sigma_factor']
    lmbda = params['lambda']
    learning_rate = params['learning_rate']
    nScales = params['nScales']
    scale_step = params['scale_step']
    scale_sigma_factor = params['scale_sigma_factor']
    scale_model_max_area = params['scale_model_max_area']

    video_path = params['video_path']
    img_files = params['img_files']
    pos = np.floor(params['init_pos'])
    target_sz = np.floor(params['target_sz'])

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
    [rs, cs] = np.meshgrid(np.arange(sz[0]) - np.floor(sz[0] / 2),
                           np.arange(sz[1]) - np.floor(sz[1] / 2))

    y = np.exp(-0.5 * ((rs**2 + cs**2) / output_sigma**2))
    yf = np.fft.fft2(y)

    # desired scale filter output (gaussian shaped), bandwidth proportional to
    # number of scales
    scale_sigma = nScales/np.sqrt(33) * scale_sigma_factor
    ss = np.arange(nScales) - np.ceil(nScales/2)
    ys = np.exp(-0.5 * (ss**2) / scale_sigma**2)
    ysf = np.fft.fft(ys)

    # store pre-computed translation filter cosine window
    # TODO: check shape. The transpose suggests sz should contain vectors
    cos_window = np.hanning(sz[0]) * np.hanning(sz[1]).reshape(-1, 1)

    # store pre-computed scale filter cosine window
    if nScales % 2 == 0:
        scale_window = np.hanning(nScales + 1)
        scale_window = scale_window[1:]
    else:
        scale_window = np.hanning(nScales)

    ss = np.arange(nScales)
    scaleFactors = scale_step**(np.ceil(nScales / 2) - ss)

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
    max_scale_factor = scale_step ** np.floor(np.log(np.minimum(im.shape[0],
                                                                im.shape[1]) /
                                                     base_target_sz) /
                                              np.log(scale_step))

    for frame in range(0, num_frames):
        # load image
        im = plt.imread(os.path.join(video_path, img_files[frame]))
        im = im.astype(np.float32)

        t1 = time.time()

        if frame > 0:
            # extract the test sample feature map for the translation filter
            xt = get_translation_sample(im,
                                        pos,
                                        sz,
                                        currentScaleFactor,
                                        cos_window)

            # calculate the correlation response of the translation filter
            xtf = np.fft.fft2(xt)
            response = np.real(np.fft.ifft2(np.sum(hf_num * xtf, 2) /
                                            (hf_den + lmbda)))

            # find the maximum translation response
            [row, col] = np.argwhere(response == np.max(response.ravel()))[0]

            # update the position
            pos += np.round((-sz/2 + np.array([row, col])) * currentScaleFactor)

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
            recovered_scale = np.argwhere(
                scale_response == np.max(scale_response.ravel()))[0]

            # update the scale
            currentScaleFactor = currentScaleFactor * \
                                 scaleFactors(recovered_scale)

            if currentScaleFactor < min_scale_factor:
                currentScaleFactor = min_scale_factor
            elif currentScaleFactor > max_scale_factor:
                currentScaleFactor = max_scale_factor

        # extract the training sample feature map for the translation filter
        xl = get_translation_sample(im, pos, sz, currentScaleFactor, cos_window)

        # calculate the translation filter update
        xlf = np.fft.fft2(xl)
        new_hf_num = yf * np.conj(xlf)
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
        positions[frame, :] = [pos, target_sz]

        tt += time.time() - t1


