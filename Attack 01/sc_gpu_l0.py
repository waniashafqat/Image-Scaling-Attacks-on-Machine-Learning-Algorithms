import sys
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import utils
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ScalingCamouflageGPU(object):
    def __init__(self, sourceImg=None, targetImg=None, **kwargs):
        _, __, *channel = sourceImg.shape
        if not channel:
            self.sourceImg = sourceImg[:, :, np.newaxis]
        else:
            self.sourceImg = sourceImg

        _, __, *channel = targetImg.shape
        if not channel:
            self.targetImg = targetImg[:, :, np.newaxis]
        else:
            self.targetImg = targetImg

        # Initialize the parameters
        self.params = {'func': cv2.resize,
                       'interpolation': cv2.INTER_LINEAR,
                       'L_dist': 'L2',
                       'penalty': 1.,
                       'img_factor': 255.}
        keys = self.params.keys()
        # Set the parameters
        for key, value in kwargs.items():
            assert key in keys, ('Improper parameter %s, '
                                 'The parameter should in: '
                                 '%s' %(key, keys))
            self.params[key] = value

    def setResizeMethod(self, func=cv2.resize,
                        interpolation=cv2.INTER_NEAREST):

        self.params['func'] = func
        self.params['interpolation'] = interpolation

    def setSourceImg(self, sourceImg):
        _, __, *channel = sourceImg.shape
        if not channel:
            self.sourceImg = sourceImg[:, :, np.newaxis]
        else:
            self.sourceImg = sourceImg

    def setTargetImg(self, targetImg):
        _, __, *channel = targetImg.shape
        if not channel:
            self.targetImg = targetImg[:, :, np.newaxis]
        else:
            self.targetImg = targetImg

    def estimateConvertMatrix(self, inSize, outSize):
           # inSize: The original input size before resizing.
           # outSize: The output size after resizing.
            # Returns an estimated convert matrix.

        # Input a dummy test image (An identity matrix * 255).
        inputDummyImg = (self.params['img_factor'] *
                         np.eye(inSize)).astype('uint8')
        outputDummyImg = self._resize(inputDummyImg,
                                      outShape=(inSize, outSize))
        # Scale the elements of convertMatrix within [0,1]
        convertMatrix = (outputDummyImg[:,:,0] /
                (np.sum(outputDummyImg[:,:,0], axis=1)).reshape(outSize, 1))

        return convertMatrix

    def _resize(self, inputImg, outShape=(0,0)):
        func = self.params['func']
        interpolation = self.params['interpolation']

        if func is Image.Image.resize:
            inputImg = Image.fromarray(inputImg)
        if func is cv2.resize:
            outputImg = func(inputImg, outShape, interpolation=interpolation)
        else:
            outputImg = func(inputImg, outShape, interpolation)
            outputImg = np.array(outputImg)
        if len(outputImg.shape) == 2:
            outputImg = outputImg[:,:,np.newaxis]
        return np.array(outputImg)

    def _getPerturbationGPU(self, convertMatrixL, convertMatrixR, source, target):
        penalty_factor = self.params['penalty']
        p, q, c = source.shape
        a, b, c = target.shape

        convertMatrixL = tf.constant(convertMatrixL, dtype=tf.float32)
        convertMatrixR = tf.constant(convertMatrixR, dtype=tf.float32)
        modifier_init = np.zeros(source.shape)
        source = tf.constant(source, dtype=tf.float32)
        target = tf.constant(target, dtype=tf.float32)
        modifier = tf.Variable(modifier_init, dtype=tf.float32)
        feasible_area = np.ones(source.shape)
        prev_obj = np.inf

        while True:
            attack = source * (1 - feasible_area) + 0.5 * (tf.tanh(modifier) + 1) * feasible_area

            x = tf.reshape(attack, [p, -1])
            x = tf.matmul(convertMatrixL, x)
            x = tf.reshape(x, [-1, q, c])
            x = tf.transpose(x, [1, 0, 2])
            x = tf.reshape(x, [q, -1])
            x = tf.matmul(convertMatrixR, x)
            x = tf.reshape(x, [-1, a, c])
            output = tf.transpose(x, [1, 0, 2])

            delta_1 = attack - source
            delta_2 = output - target

            obj1 = tf.reduce_sum(tf.square(delta_1)) / (p * q)
            obj2 = penalty_factor * tf.reduce_sum(tf.square(delta_2)) / (a * b)

            obj = obj1 + obj2

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
            with tf.GradientTape() as tape:
                tape.watch(modifier)
                attack = source * (1 - feasible_area) + 0.5 * (tf.tanh(modifier) + 1) * feasible_area
                x = tf.reshape(attack, [p, -1])
                x = tf.matmul(convertMatrixL, x)
                x = tf.reshape(x, [-1, q, c])
                x = tf.transpose(x, [1, 0, 2])
                x = tf.reshape(x, [q, -1])
                x = tf.matmul(convertMatrixR, x)
                x = tf.reshape(x, [-1, a, c])
                output = tf.transpose(x, [1, 0, 2])
                delta_1 = attack - source
                delta_2 = output - target
                obj1 = tf.reduce_sum(tf.square(delta_1)) / (p * q)
                obj2 = penalty_factor * tf.reduce_sum(tf.square(delta_2)) / (a * b)
                obj = obj1 + obj2
            gradients = tape.gradient(obj, [modifier])
            optimizer.apply_gradients(zip(gradients, [modifier]))

            obj_value = obj.numpy()
            if obj_value > 0.999 * prev_obj:
                break
            prev_obj = obj_value

        attack_opt = attack.numpy()
        return attack_opt

    def attack(self):
        sourceImg = self.sourceImg
        targetImg = self.targetImg

        sourceHeight, sourceWidth, sourceChannel = sourceImg.shape
        targetHeight, targetWidth, targetChannel = targetImg.shape

        #print(f"Source image shape: {sourceImg.shape}, min: {np.min(sourceImg)}, max: {np.max(sourceImg)}")
        #print(f"Target image shape: {targetImg.shape}, min: {np.min(targetImg)}, max: {np.max(targetImg)}")

        convertMatrixL = self.estimateConvertMatrix(sourceHeight, targetHeight)
        convertMatrixR = self.estimateConvertMatrix(sourceWidth, targetWidth)

        #print(f"convertMatrixL shape: {convertMatrixL.shape}")
        #print(f"convertMatrixR shape: {convertMatrixR.shape}")

        img_factor = self.params['img_factor']
        sourceImg = sourceImg / img_factor
        targetImg = targetImg / img_factor

        source = sourceImg
        target = targetImg
        self.info()
        attackImg = self._getPerturbationGPU(convertMatrixL,convertMatrixR, source, target)

        #perturb = perturb_opt
        #attackImg = sourceImg + perturb

        print(f"Maximum Pixel: {np.max(attackImg)}")
        print(f"Minimum Pixel: {np.min(attackImg)}")
        print('\nATTACK SUCCESSFUL!')
        return np.uint8(attackImg * img_factor)

    def info(self):
        if self.params['func'] is cv2.resize:
            func_name = 'cv2.resize'
            inter_dict = ['cv2.INTER_NEAREST',
                          'cv2.INTER_LINEAR',
                          'cv2.INTER_CUBIC',
                          'cv2.INTER_AREA',
                          'cv2.INTER_LANCZOS4']
            inter_name = inter_dict[self.params['interpolation']]
        elif self.params['func'] is Image.Image.resize:
            func_name = 'PIL.Image.resize'
            inter_dict= ['PIL.Image.NEAREST',
                         'PIL.Image.LANCZOS',
                         'PIL.Image.BILINEAR',
                         'PIL.Image.BICUBIC']
            inter_name = inter_dict[self.params['interpolation']]

        # Note: The shape read from the image
        #       matrix is (height,width,channel)!
        sourceShape = (self.sourceImg.shape[1],
                       self.sourceImg.shape[0],
                       self.sourceImg.shape[2])

        targetShape = (self.targetImg.shape[1],
                       self.targetImg.shape[0],
                       self.targetImg.shape[2])

        print("\nLet's implement an Image Scaling Attack!")
        print('Source Image Size: %s' %str(sourceShape))
        print('Target Image Size: %s' %str(targetShape))
        print('Resize Method: %s' %func_name)
        print('Interpolation Technique: %s' %inter_name)


def test():
    sourceImgPath = sys.argv[1]
    targetImgPath = sys.argv[2]
    attackImgPath = sys.argv[3]

    sourceImg = utils.imgLoader(sourceImgPath)
    targetImg = utils.imgLoader(targetImgPath)

    print("Source image: %s" %sourceImgPath)
    print("Target image: %s" %targetImgPath)

    sc_gpu = ScalingCamouflageGPU(sourceImg,
                  targetImg,
                  func=cv2.resize,
                  interpolation=cv2.INTER_LINEAR,
                  penalty=1,
                  img_factor=255.)

    attackImg = sc_gpu.attack()
    utils.imgSaver(attackImgPath, attackImg)
    print("The attack image is saved as %s" %attackImgPath)

if __name__ == '__main__':
    test()
