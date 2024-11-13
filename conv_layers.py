import numpy as np

class Convolution2D:
    def _init_(self, image, no_of_kernels, kernel_shape):
        self.image = image
        self.no_of_kernels = no_of_kernels
        self.kernel_shape = kernel_shape
        self.kernels = np.random.randn(self.no_of_kernels, self.kernel_shape[0], self.kernel_shape[1])
        self.output_size = self.image.shape[0] - self.kernel_shape[0] + 1
        self.output = []

    def forward(self):
        for kernel in self.kernels:
            result = self.filtering(self.image, kernel)
            self.output.append(result)

    def filtering(self, image, kernel):
        result_height = image.shape[0] - kernel.shape[0] + 1
        result_width = image.shape[1] - kernel.shape[1] + 1
        result = np.zeros((result_height, result_width))
        for i in range(result_height):
            for j in range(result_width):
                region = image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
                result[i, j] = np.sum(region * kernel)
        relu_result = self.Activation_ReLU(result)
        return relu_result

    def Activation_ReLU(self, regions):
        regions = np.maximum(0, regions)
        return regions


class MaxPool2D:
    def _init_(self, input_images, window_size, stride):
        self.input_images = input_images
        self.stride = stride
        self.window_size = window_size
        self.output_height = abs(self.input_images.shape[1] - self.window_size[0]) // stride + 1
        self.output_width = abs(self.input_images.shape[1] - self.window_size[1]) // stride + 1

    def apply_maxpool(self):
        self.outputs = []
        for input_image in self.input_images:
            output = np.zeros((self.output_height, self.output_width))
            for i in range(0, input_image.shape[0] - 1, self.stride):
                for j in range(0, input_image.shape[1] - 1, self.stride):
                    region = input_image[i:i + self.window_size[0], j:j + self.window_size[1]]
                    max_num = np.max(region)
                    output[i // self.stride, j // self.stride] = max_num
            self.outputs.append(output)