import numpy as np
from collections import OrderedDict
from common.layers import *


class MultimodalNet:
    def __init__(self, image_size=224*224*3, vcdr_size=1, hidden_size_list=[100, 100],
                 output_size=2, activation='relu', weight_init_std='relu',
                 weight_decay_lambda=0, use_dropout=False, dropout_ration=0.5, use_batchnorm=False):

        self.params = {}
        self.layers = OrderedDict()

        # Image Processing Networks (three parallel networks for different image types)
        self.square_net = self._create_image_network(
            'square', image_size, hidden_size_list[0], activation,
            weight_init_std, use_dropout, dropout_ration, use_batchnorm)

        self.cropped_net = self._create_image_network(
            'cropped', image_size, hidden_size_list[0], activation,
            weight_init_std, use_dropout, dropout_ration, use_batchnorm)

        self.nerve_net = self._create_image_network(
            'nerve', image_size, hidden_size_list[0], activation,
            weight_init_std, use_dropout, dropout_ration, use_batchnorm)

        # vCDR Processing Network
        self.vcdr_net = self._create_vcdr_network(
            'vcdr', vcdr_size, hidden_size_list[0]//4, activation,
            weight_init_std, use_dropout, dropout_ration, use_batchnorm)

        # Fusion Network
        # 3 images + vCDR
        fusion_input_size = hidden_size_list[0] * 3 + hidden_size_list[0]//4
        self.fusion_net = self._create_fusion_network(
            fusion_input_size, hidden_size_list[1], output_size, activation,
            weight_init_std, use_dropout, dropout_ration, use_batchnorm)

        self.weight_decay_lambda = weight_decay_lambda
        self.last_layer = SoftmaxWithLoss()

    def _create_image_network(self, prefix, input_size, hidden_size, activation,
                              weight_init_std, use_dropout, dropout_ration, use_batchnorm):
        layers = OrderedDict()

        # Initialize weights
        scale = weight_init_std
        if str(weight_init_std).lower() in ('relu', 'he'):
            scale = np.sqrt(2.0 / input_size)
        elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
            scale = np.sqrt(1.0 / input_size)

        self.params[f'W_{prefix}'] = scale * \
            np.random.randn(input_size, hidden_size)
        self.params[f'b_{prefix}'] = np.zeros(hidden_size)

        # Create layers
        layers[f'Affine_{prefix}'] = Affine(self.params[f'W_{prefix}'],
                                            self.params[f'b_{prefix}'])

        if use_batchnorm:
            self.params[f'gamma_{prefix}'] = np.ones(hidden_size)
            self.params[f'beta_{prefix}'] = np.zeros(hidden_size)
            layers[f'BatchNorm_{prefix}'] = BatchNormalization(
                self.params[f'gamma_{prefix}'], self.params[f'beta_{prefix}'])

        layers[f'Activation_{prefix}'] = Relu(
        ) if activation == 'relu' else Sigmoid()

        if use_dropout:
            layers[f'Dropout_{prefix}'] = Dropout(dropout_ration)

        return layers

    def _create_vcdr_network(self, prefix, input_size, hidden_size, activation,
                             weight_init_std, use_dropout, dropout_ration, use_batchnorm):
        return self._create_image_network(prefix, input_size, hidden_size, activation,
                                          weight_init_std, use_dropout, dropout_ration, use_batchnorm)

    def _create_fusion_network(self, input_size, hidden_size, output_size, activation,
                               weight_init_std, use_dropout, dropout_ration, use_batchnorm):
        layers = OrderedDict()

        # Hidden layer
        scale = np.sqrt(
            2.0 / input_size) if activation == 'relu' else np.sqrt(1.0 / input_size)
        self.params['W_fusion1'] = scale * \
            np.random.randn(input_size, hidden_size)
        self.params['b_fusion1'] = np.zeros(hidden_size)

        layers['Affine_fusion1'] = Affine(
            self.params['W_fusion1'], self.params['b_fusion1'])

        if use_batchnorm:
            self.params['gamma_fusion'] = np.ones(hidden_size)
            self.params['beta_fusion'] = np.zeros(hidden_size)
            layers['BatchNorm_fusion'] = BatchNormalization(
                self.params['gamma_fusion'], self.params['beta_fusion'])

        layers['Activation_fusion'] = Relu(
        ) if activation == 'relu' else Sigmoid()

        if use_dropout:
            layers['Dropout_fusion'] = Dropout(dropout_ration)

        # Output layer
        self.params['W_fusion2'] = scale * \
            np.random.randn(hidden_size, output_size)
        self.params['b_fusion2'] = np.zeros(output_size)
        layers['Affine_fusion2'] = Affine(
            self.params['W_fusion2'], self.params['b_fusion2'])

        return layers

    def predict(self, x_square, x_cropped, x_nerve, x_vcdr, train_flg=False):
        # Process each input stream
        h_square = x_square
        h_cropped = x_cropped
        h_nerve = x_nerve
        h_vcdr = x_vcdr

        # Forward through image networks
        for key, layer in self.square_net.items():
            h_square = self._forward_layer(layer, h_square, train_flg, key)

        for key, layer in self.cropped_net.items():
            h_cropped = self._forward_layer(layer, h_cropped, train_flg, key)

        for key, layer in self.nerve_net.items():
            h_nerve = self._forward_layer(layer, h_nerve, train_flg, key)

        # Forward through vCDR network
        for key, layer in self.vcdr_net.items():
            h_vcdr = self._forward_layer(layer, h_vcdr, train_flg, key)

        # Concatenate features
        h = np.concatenate([h_square, h_cropped, h_nerve, h_vcdr], axis=1)

        # Forward through fusion network
        for key, layer in self.fusion_net.items():
            h = self._forward_layer(layer, h, train_flg, key)

        return h

    def _forward_layer(self, layer, x, train_flg, key):
        if "Dropout" in key or "BatchNorm" in key:
            return layer.forward(x, train_flg)
        return layer.forward(x)

    def loss(self, x_square, x_cropped, x_nerve, x_vcdr, t, train_flg=False):
        y = self.predict(x_square, x_cropped, x_nerve, x_vcdr, train_flg)

        # Weight decay
        weight_decay = 0
        for key, W in self.params.items():
            if key.startswith('W'):
                weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x_square, x_cropped, x_nerve, x_vcdr, t):
        y = self.predict(x_square, x_cropped, x_nerve, x_vcdr, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x_square.shape[0])
        return accuracy

    def gradient(self, x_square, x_cropped, x_nerve, x_vcdr, t):
        # forward
        self.loss(x_square, x_cropped, x_nerve, x_vcdr, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        # Fusion network backward
        layers = list(self.fusion_net.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Split gradients for each stream
        d_size = self.params['W_square'].shape[1]
        d_vcdr_size = self.params['W_vcdr'].shape[1]

        d_square = dout[:, :d_size]
        d_cropped = dout[:, d_size:2*d_size]
        d_nerve = dout[:, 2*d_size:3*d_size]
        d_vcdr = dout[:, 3*d_size:]

        # Backward through each stream
        for layer in reversed(list(self.square_net.values())):
            d_square = layer.backward(d_square)

        for layer in reversed(list(self.cropped_net.values())):
            d_cropped = layer.backward(d_cropped)

        for layer in reversed(list(self.nerve_net.values())):
            d_nerve = layer.backward(d_nerve)

        for layer in reversed(list(self.vcdr_net.values())):
            d_vcdr = layer.backward(d_vcdr)

        # Collect gradients
        grads = {}
        for key in self.params.keys():
            if key.startswith('W_'):
                # Handle individual network weights
                if 'square' in key:
                    layer = self.square_net[f'Affine_{key.split("_")[1]}']
                    grads[key] = layer.dW + \
                        self.weight_decay_lambda * self.params[key]
                elif 'cropped' in key:
                    layer = self.cropped_net[f'Affine_{key.split("_")[1]}']
                    grads[key] = layer.dW + \
                        self.weight_decay_lambda * self.params[key]
                elif 'nerve' in key:
                    layer = self.nerve_net[f'Affine_{key.split("_")[1]}']
                    grads[key] = layer.dW + \
                        self.weight_decay_lambda * self.params[key]
                elif 'vcdr' in key:
                    layer = self.vcdr_net[f'Affine_{key.split("_")[1]}']
                    grads[key] = layer.dW + \
                        self.weight_decay_lambda * self.params[key]
                # Handle fusion network weights
                elif key == 'W_fusion1':
                    layer = self.fusion_net['Affine_fusion1']
                    grads[key] = layer.dW + \
                        self.weight_decay_lambda * self.params[key]
                elif key == 'W_fusion2':
                    layer = self.fusion_net['Affine_fusion2']
                    grads[key] = layer.dW + \
                        self.weight_decay_lambda * self.params[key]

            elif key.startswith('b_'):
                # Handle individual network biases
                if 'square' in key:
                    layer = self.square_net[f'Affine_{key.split("_")[1]}']
                    grads[key] = layer.db
                elif 'cropped' in key:
                    layer = self.cropped_net[f'Affine_{key.split("_")[1]}']
                    grads[key] = layer.db
                elif 'nerve' in key:
                    layer = self.nerve_net[f'Affine_{key.split("_")[1]}']
                    grads[key] = layer.db
                elif 'vcdr' in key:
                    layer = self.vcdr_net[f'Affine_{key.split("_")[1]}']
                    grads[key] = layer.db
                # Handle fusion network biases
                elif key == 'b_fusion1':
                    layer = self.fusion_net['Affine_fusion1']
                    grads[key] = layer.db
                elif key == 'b_fusion2':
                    layer = self.fusion_net['Affine_fusion2']
                    grads[key] = layer.db

            elif key.startswith('gamma_'):
                # Handle BatchNorm gamma gradients
                if 'square' in key:
                    layer = self.square_net[f'BatchNorm_{key.split("_")[1]}']
                    grads[key] = layer.dgamma
                elif 'cropped' in key:
                    layer = self.cropped_net[f'BatchNorm_{key.split("_")[1]}']
                    grads[key] = layer.dgamma
                elif 'nerve' in key:
                    layer = self.nerve_net[f'BatchNorm_{key.split("_")[1]}']
                    grads[key] = layer.dgamma
                elif 'vcdr' in key:
                    layer = self.vcdr_net[f'BatchNorm_{key.split("_")[1]}']
                    grads[key] = layer.dgamma
                elif key == 'gamma_fusion':
                    layer = self.fusion_net['BatchNorm_fusion']
                    grads[key] = layer.dgamma

            elif key.startswith('beta_'):
                # Handle BatchNorm beta gradients
                if 'square' in key:
                    layer = self.square_net[f'BatchNorm_{key.split("_")[1]}']
                    grads[key] = layer.dbeta
                elif 'cropped' in key:
                    layer = self.cropped_net[f'BatchNorm_{key.split("_")[1]}']
                    grads[key] = layer.dbeta
                elif 'nerve' in key:
                    layer = self.nerve_net[f'BatchNorm_{key.split("_")[1]}']
                    grads[key] = layer.dbeta
                elif 'vcdr' in key:
                    layer = self.vcdr_net[f'BatchNorm_{key.split("_")[1]}']
                    grads[key] = layer.dbeta
                elif key == 'beta_fusion':
                    layer = self.fusion_net['BatchNorm_fusion']
                    grads[key] = layer.dbeta

        return grads
