import numpy as np
from models.deep_convnet import Convolution, Pooling, Relu, Affine, SoftmaxWithLoss


class MultiscaleConvNet:
    def __init__(self, input_dim=(3, 224, 224), num_classes=10):
        # 기존 구조는 그대로 유지
        self.conv1_3x3 = Convolution(np.random.randn(16, 3, 3, 3), np.zeros(16), stride=1, pad=1)
        self.conv1_5x5 = Convolution(np.random.randn(16, 3, 5, 5), np.zeros(16), stride=1, pad=2)
        self.conv1_7x7 = Convolution(np.random.randn(16, 3, 7, 7), np.zeros(16), stride=1, pad=3)
        self.relu1 = Relu()
        self.pool1 = Pooling(2, 2, stride=2)  # 224 -> 112

        # Conv2 + Pool2
        self.conv2 = Convolution(np.random.randn(32, 48, 3, 3), np.zeros(32), stride=1, pad=1)
        self.relu2 = Relu()
        self.pool2 = Pooling(2, 2, stride=2)  # 112 -> 56

        # Conv3 + Pool3
        self.conv3 = Convolution(np.random.randn(64, 32, 3, 3), np.zeros(64), stride=1, pad=1)
        self.relu3 = Relu()
        self.pool3 = Pooling(2, 2, stride=2)  # 56 -> 28

        # Conv4 + Pool4
        self.conv4 = Convolution(np.random.randn(128, 64, 3, 3), np.zeros(128), stride=1, pad=1)
        self.relu4 = Relu()
        self.pool4 = Pooling(2, 2, stride=2)  # 28 -> 14

        # Conv5 + Pool5
        self.conv5 = Convolution(np.random.randn(256, 128, 3, 3), np.zeros(256), stride=1, pad=1)
        self.relu5 = Relu()
        self.pool5 = Pooling(2, 2, stride=2)  # 14 -> 7

        # Fully connected layers
        self.affine1 = Affine(np.random.randn(256 * 7 * 7, 128), np.zeros(128))
        self.affine2 = Affine(np.random.randn(128, num_classes), np.zeros(num_classes))

        # Softmax with loss layer 추가
        self.last_layer = SoftmaxWithLoss()

        # params 딕셔너리 추가 - 각 레이어의 W와 b를 직접 참조
        self.params = {
            'W1': self.conv1_3x3.W, 'b1': self.conv1_3x3.b,
            'W2': self.conv1_5x5.W, 'b2': self.conv1_5x5.b,
            'W3': self.conv1_7x7.W, 'b3': self.conv1_7x7.b,
            'W4': self.conv2.W, 'b4': self.conv2.b,
            'W5': self.conv3.W, 'b5': self.conv3.b,
            'W6': self.conv4.W, 'b6': self.conv4.b,
            'W7': self.conv5.W, 'b7': self.conv5.b,
            'W8': self.affine1.W, 'b8': self.affine1.b,
            'W9': self.affine2.W, 'b9': self.affine2.b
        }

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

        # Flatten output for the fully connected layers
        pool_out = pool_out.reshape(pool_out.shape[0], -1)

        affine_out = self.affine1.forward(pool_out)
        score = self.affine2.forward(affine_out)

        return score

    def backward(self, dout):
        # Last layer -> Affine2 -> Affine1 역전파
        dout = self.affine2.backward(dout)
        dout = self.affine1.backward(dout)

        # Affine layer에서 나온 dout을 다시 4D tensor로 reshape
        dout = dout.reshape(dout.shape[0], 256, 7, 7)

        dout = self.pool5.backward(dout)
        dout = self.relu5.backward(dout)
        dout = self.conv5.backward(dout)

        dout = self.pool4.backward(dout)
        dout = self.relu4.backward(dout)
        dout = self.conv4.backward(dout)

        dout = self.pool3.backward(dout)
        dout = self.relu3.backward(dout)
        dout = self.conv3.backward(dout)

        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)

        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)

        # Concatenated 레이어를 다시 분할
        dout_3x3, dout_5x5, dout_7x7 = np.split(dout, 3, axis=1)

        # 각 합성곱 레이어로 역전파
        dout_3x3 = self.conv1_3x3.backward(dout_3x3)
        dout_5x5 = self.conv1_5x5.backward(dout_5x5)
        dout_7x7 = self.conv1_7x7.backward(dout_7x7)

        return dout_3x3 + dout_5x5 + dout_7x7

    def gradient(self, x, t):
        # Forward
        y = self.forward(x)
        loss = self.last_layer.forward(y, t)

        # Backward
        dout = self.last_layer.backward()
        self.backward(dout)

        # 그래디언트 저장
        grads = {
            'W1': self.conv1_3x3.dW, 'b1': self.conv1_3x3.db,
            'W2': self.conv1_5x5.dW, 'b2': self.conv1_5x5.db,
            'W3': self.conv1_7x7.dW, 'b3': self.conv1_7x7.db,
            'W4': self.conv2.dW, 'b4': self.conv2.db,
            'W5': self.conv3.dW, 'b5': self.conv3.db,
            'W6': self.conv4.dW, 'b6': self.conv4.db,
            'W7': self.conv5.dW, 'b7': self.conv5.db,
            'W8': self.affine1.dW, 'b8': self.affine1.db,
            'W9': self.affine2.dW, 'b9': self.affine2.db
        }

        return grads
    
    def loss(self, x, t):
        y = self.forward(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        import time

        start_time = time.time()
        # Forward를 통해 예측값 계산
        y = self.forward(x)
        print(f"Forward 실행 시간: {time.time() - start_time:.2f} 초")

        # 예측값 중 가장 높은 확률을 가진 클래스 선택
        y = np.argmax(y, axis=1)

        # 레이블이 원-핫 인코딩된 경우 처리
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        end_time = time.time()
        print(f"Forward 및 argmax 실행 시간: {end_time - start_time:.2f} 초")

        # 정확도 계산
        accuracy = np.sum(y == t) / float(x.shape[0])

        print(f"Accuracy 계산 완료: {accuracy}")
        return accuracy
