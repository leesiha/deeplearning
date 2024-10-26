import numpy as np
from models.deep_convnet import Convolution, Pooling, Relu, Affine, SoftmaxWithLoss


class MultiscaleConvNet:
    def __init__(self, input_dim=(3, 512, 512), num_classes=10):
        # 기존 구조는 그대로 유지
        self.conv1_3x3 = Convolution(np.random.randn(
            16, 3, 3, 3), np.zeros(16), stride=1, pad=1)
        self.conv1_5x5 = Convolution(np.random.randn(
            16, 3, 5, 5), np.zeros(16), stride=1, pad=2)
        self.conv1_7x7 = Convolution(np.random.randn(
            16, 3, 7, 7), np.zeros(16), stride=1, pad=3)
        self.relu1 = Relu()
        self.pool1 = Pooling(2, 2, stride=2)  # 512 -> 256

        # Conv2 + Pool2
        self.conv2 = Convolution(np.random.randn(
            32, 48, 3, 3), np.zeros(32), stride=1, pad=1)
        self.relu2 = Relu()
        self.pool2 = Pooling(2, 2, stride=2)  # 256 -> 128

        # Conv3 + Pool3
        self.conv3 = Convolution(np.random.randn(
            64, 32, 3, 3), np.zeros(64), stride=1, pad=1)
        self.relu3 = Relu()
        self.pool3 = Pooling(2, 2, stride=2)  # 128 -> 64

        # Conv4 + Pool4
        self.conv4 = Convolution(np.random.randn(
            128, 64, 3, 3), np.zeros(128), stride=1, pad=1)
        self.relu4 = Relu()
        self.pool4 = Pooling(2, 2, stride=2)  # 64 -> 32

        # Conv5 + Pool5
        self.conv5 = Convolution(np.random.randn(
            256, 128, 3, 3), np.zeros(256), stride=1, pad=1)
        self.relu5 = Relu()
        self.pool5 = Pooling(2, 2, stride=2)  # 32 -> 16

        # Conv6 + 마지막 풀링 레이어 추가
        self.conv6 = Convolution(np.random.randn(
            512, 256, 3, 3), np.zeros(512), stride=1, pad=1)  # 추가된 Conv
        self.relu6 = Relu()
        self.pool6 = Pooling(2, 2, stride=2)  # 16 -> 8 (마지막)

        # Fully connected layers
        self.affine1 = Affine(np.random.randn(
            512 * 8 * 8, 128), np.zeros(128))  # 8x8 출력에 맞춤
        self.affine2 = Affine(np.random.randn(
            128, num_classes), np.zeros(num_classes))
        self.last_layer = SoftmaxWithLoss()

    def forward(self, x):
        # Multiscale convolution
        conv_3x3 = self.conv1_3x3.forward(x)
        conv_5x5 = self.conv1_5x5.forward(x)
        conv_7x7 = self.conv1_7x7.forward(x)

        # Concatenate outputs
        concat = np.concatenate((conv_3x3, conv_5x5, conv_7x7), axis=1)

        # First pooling
        relu_out = self.relu1.forward(concat)
        pool_out = self.pool1.forward(relu_out)

        # Second conv and pooling
        conv_out = self.conv2.forward(pool_out)
        relu_out = self.relu2.forward(conv_out)
        pool_out = self.pool2.forward(relu_out)

        # Third conv and pooling
        conv_out = self.conv3.forward(pool_out)
        relu_out = self.relu3.forward(conv_out)
        pool_out = self.pool3.forward(relu_out)

        # Fourth conv and pooling
        conv_out = self.conv4.forward(pool_out)
        relu_out = self.relu4.forward(conv_out)
        pool_out = self.pool4.forward(relu_out)

        # Fifth conv and pooling
        conv_out = self.conv5.forward(pool_out)
        relu_out = self.relu5.forward(conv_out)
        pool_out = self.pool5.forward(relu_out)

        # Conv6 + 마지막 풀링
        conv_out = self.conv6.forward(pool_out)
        relu_out = self.relu6.forward(conv_out)
        pool_out = self.pool6.forward(relu_out)

        # Flatten output for the fully connected layers
        pool_out = pool_out.reshape(pool_out.shape[0], -1)
        print(f"After flattening: {pool_out.shape}")  # 평탄화 후 크기 확인

        affine_out = self.affine1.forward(pool_out)
        score = self.affine2.forward(affine_out)

        return score


    def loss(self, x, t):
        y = self.forward(x)
        return self.last_layer.forward(y, t)

    def backward(self, dout=1):
        # Backward pass (Backpropagation)
        dout = self.last_layer.backward(dout)
        dout = self.affine2.backward(dout)
        dout = self.affine1.backward(dout)
        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)

        # Multiscale Convolution layers backpropagation
        dout_3x3 = self.conv1_3x3.backward(dout)
        dout_5x5 = self.conv1_5x5.backward(dout)
        dout_7x7 = self.conv1_7x7.backward(dout)

        return dout_3x3 + dout_5x5 + dout_7x7

    def gradient(self, x, t):
        # Forward pass and compute loss
        self.loss(x, t)

        # Backward pass to compute gradients
        dout = 1  # Assuming softmax with cross-entropy loss
        self.backward(dout)

        # Store gradients in a dictionary
        grads = {}
        grads['W1'], grads['b1'] = self.conv1_3x3.dW, self.conv1_3x3.db
        grads['W2'], grads['b2'] = self.conv1_5x5.dW, self.conv1_5x5.db
        grads['W3'], grads['b3'] = self.conv1_7x7.dW, self.conv1_7x7.db
        grads['W4'], grads['b4'] = self.conv2.dW, self.conv2.db
        grads['W5'], grads['b5'] = self.conv3.dW, self.conv3.db  # 추가된 레이어 예시
        grads['W6'], grads['b6'] = self.affine1.dW, self.affine1.db  # 전결합층

        return grads
