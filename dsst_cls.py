import numpy as np

import skimage.feature as skif
import skimage.transform as skit
import skimage.color as skic
import skimage.io as skio
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


class DSSTTracker(object):

    def __init__(self,
                 im_height,
                 im_width,
                 padding=1,
                 output_sigma_factor=1/16.0,
                 lmbda=0.01,
                 learning_rate=0.025,
                 n_scales=33,
                 scale_step=1.02,
                 scale_sigma_factor=1/4.0,
                 scale_model_max_area=512,
                 target_sz=np.array([36, 51]),
                 visualization=False):

        self.padding = padding
        self.output_sigma_factor = output_sigma_factor
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.n_scales = n_scales
        self.scale_step = scale_step
        self.scale_sigma_factor = scale_sigma_factor
        self.scale_model_max_area = scale_model_max_area
        self.base_target_sz = np.floor(np.asarray(target_sz))
        self.visualization = visualization

        # window size, taking padding into account
        self.sz = np.floor(self.base_target_sz * (1 + padding))

        self.yf = self.spatial_filter_output_prototype()
        self.ysf = self.scale_filter_output_prototype()
        self.cos_window = self.compute_spatial_filter_window()
        self.scale_window = self.compute_scale_filter_window()
        self.scale_factors = self.compute_scale_factors()

        self.current_scale_factor = 1
        self.scale_model_sz = self.compute_resize_dimensions(target_sz)
        self.min_scale_factor, self.max_scale_factor = \
            self.compute_min_max_scales(im_height, im_width)

        self.hf_den = None
        self.hf_num = None
        self.sf_den = None
        self.sf_num = None
        self.pos = None

    def spatial_filter_output_prototype(self):

        # desired translation filter output (gaussian shaped), bandwidth
        # proportional to target size
        output_sigma = np.sqrt(np.prod(self.base_target_sz)) * \
                       self.output_sigma_factor
        [rs, cs] = np.meshgrid(np.arange(1, self.sz[0] + 1)
                               - np.floor(self.sz[0] / 2),
                               np.arange(1, self.sz[1] + 1)
                               - np.floor(self.sz[1] / 2),
                               indexing='ij')

        y = np.exp(-0.5 * ((rs ** 2 + cs ** 2) / output_sigma ** 2))
        yf = np.fft.fft2(y, axes=(0, 1))

        return yf

    def scale_filter_output_prototype(self):
        # desired scale filter output (gaussian shaped), bandwidth proportional to
        # number of scales
        scale_sigma = self.n_scales / np.sqrt(33) * self.scale_sigma_factor
        ss = np.arange(self.n_scales) - np.ceil(self.n_scales / 2)
        ys = np.exp(-0.5 * (ss ** 2) / scale_sigma ** 2)
        ysf = np.fft.fft(ys)

        return ysf

    def compute_spatial_filter_window(self):
        cos_window = np.hanning(self.sz[0]).reshape(-1, 1) * \
                     np.hanning(self.sz[1])

        return cos_window

    def compute_scale_filter_window(self):
        if self.n_scales % 2 == 0:
            scale_window = np.hanning(self.n_scales + 1)
            scale_window = scale_window[1:]
        else:
            scale_window = np.hanning(self.n_scales)

        return scale_window

    def compute_scale_factors(self):
        ss = np.arange(self.n_scales)
        scaleFactors = self.scale_step ** (np.ceil(self.n_scales / 2) - (ss + 1))

        return scaleFactors

    def compute_resize_dimensions(self, init_target_sz):
        # compute the resize dimensions used for feature extraction in the scale
        # estimation

        scale_model_factor = 1

        if np.prod(init_target_sz) > self.scale_model_max_area:
            scale_model_factor = np.sqrt(self.scale_model_max_area /
                                         np.prod(init_target_sz))

        scale_model_sz = np.floor(init_target_sz * scale_model_factor)

        return scale_model_sz

    def compute_min_max_scales(self, im_height, im_width):
        # find maximum and minimum scales
        min_scale_factor = self.scale_step ** \
                           np.ceil(np.log(np.max(5 / self.sz)) /
                                                 np.log(self.scale_step))
        max_scale_factor = self.scale_step ** \
                           np.floor(np.log(np.min(np.array([im_height,
                                                            im_width]) /
                                                  self.base_target_sz)) /
                                    np.log(self.scale_step))

        return min_scale_factor, max_scale_factor

    def update(self, im, pos, current_scale_factor, init_tracker=False):
        # extract the training sample feature map for the translation filter
        xl = get_translation_sample(im,
                                    pos,
                                    self.sz,
                                    self.current_scale_factor,
                                    self.cos_window)

        # calculate the translation filter update
        xlf = np.fft.fft2(xl, axes=(0, 1))
        new_hf_num = repeat_to_third_dim(self.yf) * np.conj(xlf)
        new_hf_den = np.sum(xlf * np.conj(xlf), axis=2)

        # extract the training sample feature map for the scale filter
        xs = get_scale_sample(im,
                              pos,
                              self.base_target_sz,
                              current_scale_factor * self.scale_factors,
                              self.scale_window,
                              self.scale_model_sz)

        # calculate the scale filter update
        xsf = np.fft.fft(xs, axis=1)
        new_sf_num = self.ysf * np.conj(xsf)
        new_sf_den = np.sum(xsf * np.conj(xsf), axis=0)

        if self.hf_den is None or init_tracker:
            self.hf_den = new_hf_den
            self.hf_num = new_hf_num

            self.sf_den = new_sf_den
            self.sf_num = new_sf_num
        else:
            # subsequent frames, update the model
            self.hf_den = (1 - self.learning_rate) * self.hf_den \
                          + self.learning_rate * new_hf_den
            self.hf_num = (1 - self.learning_rate) * self.hf_num \
                          + self.learning_rate * new_hf_num
            self.sf_den = (1 - self.learning_rate) * self.sf_den \
                          + self.learning_rate * new_sf_den
            self.sf_num = (1 - self.learning_rate) * self.sf_num \
                          + self.learning_rate * new_sf_num

    def predict(self, im, pos):
        # extract the test sample feature map for the translation filter
        xt = get_translation_sample(im,
                                    pos,
                                    self.sz,
                                    self.current_scale_factor,
                                    self.cos_window)

        # calculate the correlation response of the translation filter
        xtf = np.fft.fft2(xt, axes=(0, 1))
        response = np.real(np.fft.ifft2(np.sum(self.hf_num * xtf, 2) /
                                        (self.hf_den + self.lmbda), axes=(0, 1)
                                        ))
        # find the maximum translation response
        [row, col] = np.argwhere(response == np.max(response.ravel()))[0]

        # update the position
        pos += np.round(
            (-self.sz / 2 + np.array([row, col]) + 1) * self.current_scale_factor)

        # extract the test sample feature map for the scale filter
        xs = get_scale_sample(im,
                              pos,
                              self.base_target_sz,
                              self.current_scale_factor * self.scale_factors,
                              self.scale_window,
                              self.scale_model_sz)

        # calculate the correlation response of the scale filter
        xsf = np.fft.fft(xs, axis=1)
        scale_response = np.real(np.fft.ifft(np.sum(self.sf_num * xsf, axis=0) /
                                             (self.sf_den + self.lmbda)))

        # find the maximum scale response
        # TODO find out why the -1 hack produces better scale estimates
        recovered_scale = np.argwhere(
            scale_response == np.max(scale_response.ravel()))[0] - 1

        # update the scale
        current_scale_factor = self.current_scale_factor * \
                               self.scale_factors[recovered_scale]

        if current_scale_factor < self.min_scale_factor:
            current_scale_factor = self.min_scale_factor
        elif current_scale_factor > self. max_scale_factor:
            current_scale_factor = self.max_scale_factor

        return pos, current_scale_factor, response, scale_response

    def track(self, im):
        self.pos, self.current_scale_factor, response, scale_response = \
            self.predict(im, self.pos)

        self.update(im, self.pos, self.current_scale_factor)

        return self.pos.copy(), self.current_scale_factor

    def initialise_tracker(self, im, pos):
        self.update(im, pos, current_scale_factor=1, init_tracker=True)
        self.pos = pos


def track_sequence(img_seq, init_pos):
    import time

    img = skio.imread(img_seq[0])
    tracker = DSSTTracker(im_height=img.shape[0],
                          im_width=img.shape[1])

    tracker.initialise_tracker(img, init_pos)

    pos_list = []
    scale_list = []

    t1 = time.time()
    for i, img_path in enumerate(img_seq[1:]):
        img = skio.imread(img_path)
        pos, scale_factor = tracker.track(img)
        pos_list += [pos]
        scale_list += [scale_factor]

        print(i, pos, scale_factor, len(pos_list) / (time.time() - t1))

    return pos_list, scale_list


if __name__ == "__main__":
    import glob

    img_seq = glob.glob('sequences/dog1/imgs/*.jpg')
    init_pos = np.array([112, 139]) + np.array([36, 51]) / 2

    pos_list, scale_list = track_sequence(img_seq, init_pos)

    # print(zip(enumerate(len(pos_list)), pos_list, scale_list))






















