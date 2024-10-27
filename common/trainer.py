# coding: utf-8
from common.optimizer import *
import numpy as np
import sys
import os
sys.path.append(os.pardir)


class Trainer:
    """신경망 훈련을 대신 해주는 클래스
    멀티모달 네트워크를 위해 수정됨: 여러 입력(이미지들과 vCDR)을 처리할 수 있음
    """

    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr': 0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose

        # Unpack the training inputs
        self.x_train_square, self.x_train_cropped, self.x_train_nerve, self.x_train_vcdr = x_train
        self.t_train = t_train

        # Unpack the test inputs
        self.x_test_square, self.x_test_cropped, self.x_test_nerve, self.x_test_vcdr = x_test
        self.t_test = t_test

        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimizer
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'nesterov': Nesterov,
                                'adagrad': AdaGrad, 'rmsprpo': RMSprop, 'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](
            **optimizer_param)

        self.train_size = self.x_train_square.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)

        # Get batches for all inputs
        x_batch_square = self.x_train_square[batch_mask]
        x_batch_cropped = self.x_train_cropped[batch_mask]
        x_batch_nerve = self.x_train_nerve[batch_mask]
        x_batch_vcdr = self.x_train_vcdr[batch_mask]
        t_batch = self.t_train[batch_mask]

        # Calculate gradients
        grads = self.network.gradient(x_batch_square, x_batch_cropped,
                                      x_batch_nerve, x_batch_vcdr, t_batch)

        # Update parameters
        self.optimizer.update(self.network.params, grads)

        # Calculate loss
        loss = self.network.loss(x_batch_square, x_batch_cropped,
                                 x_batch_nerve, x_batch_vcdr, t_batch)
        self.train_loss_list.append(loss)

        if self.current_iter % int(self.iter_per_epoch) == 0:
            self.current_epoch += 1

            # Prepare sample data for evaluation
            if self.evaluate_sample_num_per_epoch is not None:
                t = self.evaluate_sample_num_per_epoch
                x_train_square_sample = self.x_train_square[:t]
                x_train_cropped_sample = self.x_train_cropped[:t]
                x_train_nerve_sample = self.x_train_nerve[:t]
                x_train_vcdr_sample = self.x_train_vcdr[:t]
                t_train_sample = self.t_train[:t]

                x_test_square_sample = self.x_test_square[:t]
                x_test_cropped_sample = self.x_test_cropped[:t]
                x_test_nerve_sample = self.x_test_nerve[:t]
                x_test_vcdr_sample = self.x_test_vcdr[:t]
                t_test_sample = self.t_test[:t]
            else:
                x_train_square_sample = self.x_train_square
                x_train_cropped_sample = self.x_train_cropped
                x_train_nerve_sample = self.x_train_nerve
                x_train_vcdr_sample = self.x_train_vcdr
                t_train_sample = self.t_train

                x_test_square_sample = self.x_test_square
                x_test_cropped_sample = self.x_test_cropped
                x_test_nerve_sample = self.x_test_nerve
                x_test_vcdr_sample = self.x_test_vcdr
                t_test_sample = self.t_test

            # Calculate accuracy
            train_acc = self.network.accuracy(x_train_square_sample,
                                              x_train_cropped_sample,
                                              x_train_nerve_sample,
                                              x_train_vcdr_sample,
                                              t_train_sample)

            test_acc = self.network.accuracy(x_test_square_sample,
                                             x_test_cropped_sample,
                                             x_test_nerve_sample,
                                             x_test_vcdr_sample,
                                             t_test_sample)

            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print(f"=== epoch:{self.current_epoch}, train acc:"
                      f"{train_acc:.4f}, test acc:{test_acc:.4f} ===")

        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()


        test_acc = self.network.accuracy(self.x_test_square,
                                         self.x_test_cropped,
                                         self.x_test_nerve,
                                         self.x_test_vcdr,
                                         self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print(f"test acc: {test_acc:.4f}")

        return test_acc
