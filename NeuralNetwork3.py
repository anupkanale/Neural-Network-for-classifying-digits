import sys
import numpy as np
import time


class softmax_cross_entropy:
    def __init__(self):
        self.oneHot_Y = None
        self.prob = None

    def compute_loss(self, A, Y):
        # one-hot encoded (matrix) form of ground truth
        self.oneHot_Y = np.zeros(A.shape)
        self.oneHot_Y[np.arange(Y.size), Y.T] = 1.0

        # exponentials could blow up, so use following trick to stabilize
        shiftA = A - np.amax(A, axis = 1, keepdims = True)
        sum_shiftA = np.sum(np.exp(shiftA), axis = 1, keepdims = True)
        self.prob = np.exp(shiftA) / sum_shiftA

        cross_entropy_loss = np.multiply(self.oneHot_Y, shiftA - np.log(sum_shiftA))
        cross_entropy_loss = -np.sum(cross_entropy_loss)/A.shape[0]
        return cross_entropy_loss

    def compute_lossGrad(self, A):
        lossGrad = - (self.oneHot_Y - self.prob) / A.shape[0]
        return lossGrad


class linear_layer:
    def __init__(self, input_dim, output_dim):
        """
        Initialize weights and gradients

        :param input_dim: input dimension
        :param output_dim: output dimension
        """

        self.params = dict()
        self.params['W'] = np.random.normal(0, 0.1, size=[input_dim, output_dim])
        self.params['b'] = np.random.normal(0, 0.1, size=[1, output_dim])

        self.gradient = dict()
        self.gradient['W'] = np.zeros((input_dim, output_dim))
        self.gradient['b'] = np.zeros((1, output_dim))

    def forward(self, X):
        """
        Forward pass of linear layer

        :param: input array, X
        :return wx + b
        """
        return np.dot(X, self.params['W']) + self.params['b']

    def backward(self, X, grad):
        """
            The backward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_dim numpy array, the input to the forward pass.
            - grad: A N-by-output_dim numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss
                 w.r.t. forward_output[i].

            Operation:
            - Compute the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['W'], self.params['b'].

            Return:
            - backward_output: A N-by-input_dim numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss w.r.t. X[i].
        """
        self.gradient['W'] = np.dot(X.T, grad)
        self.gradient['b'] = np.sum(grad, axis=0, keepdims=True)
        backward_output = np.dot(grad, self.params['W'].T)

        return backward_output


class relu:
    def forward(self, X):
        """
        The forward pass for the RELU activation function

        :param: input array, X.
        :return: relu(x)
        """
        return np.maximum(0, X)

    def backward(self, X, grad):
        """
        The forward pass for the RELU activation function

        :param: input array of forward pass, X.
        :return: relu(x) = max(0, X)
        """
        relu_grad = (X > 0) * grad
        return relu_grad


def load_mnist_data():
    train_x = np.genfromtxt('train_image.csv', delimiter=',', dtype=int)
    train_y = np.genfromtxt('train_label.csv', delimiter=',', dtype=int)
    train_x = train_x/255 # normalize

    # Choose subset of size for training
    N = 10000
    subset_indices = np.random.choice(train_x.shape[0], N, replace=False)
    train_x, train_y = train_x[subset_indices], train_y[subset_indices]

    idx = int(1/6 * N)
    val_x, val_y = train_x[N-idx:, :], train_y[N-idx:]
    train_x, train_y = train_x[:N-idx, :], train_y[:N-idx]

    # print("from csv: ", train_x.shape, train_y.shape, val_x.shape, val_y.shape)
    return train_x, train_y, val_x, val_y


def miniBatchSGD(neuralNet, lambd, alpha):
    """
    Update weights and biases using gradient descent

    :param neuralNet dictionary containing parameters
    :param lambd: regularization coefficient
    :param alpha: learning rate

    :return neuralNet with updated parameter values
    """
    for module_name, module in neuralNet.items():
        # check if a module has learnable parameters
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                update = module.gradient[key] + lambd * module.params[key]
                module.params[key] -= alpha * update

    return neuralNet


def trainNetwork(nn_params):
    # np.random.seed(42)
    
    # Load data
    #--------------------------
    Xtrain, Ytrain, Xval, Yval = load_mnist_data()
    M_train, D = Xtrain.shape  # number of training examples, dimension
    M_val, _ = Xval.shape  # number of training examples

    # Set parameters
    #--------------------------
    neuralNet = dict()
    D_l1 = int(nn_params['nodes_l1'])  # number of nodes in hidden layer 1
    D_l2 = int(nn_params['nodes_l2'])  # number of nodes in hidden layer 2
    num_epoch = int(nn_params['num_epoch'])
    minibatch_size = int(nn_params['minibatch_size'])
    alpha = float(nn_params['alpha'])
    period = 10  # change learning rate every period number of epochs
    lambd = float(nn_params['lambda'])
    activation_func = relu

    # Initialize objects
    neuralNet['L1'] = linear_layer(D, D_l1)
    neuralNet['nonlinear1'] = activation_func()
    neuralNet['L2'] = linear_layer(D_l1, D_l2)
    neuralNet['loss'] = softmax_cross_entropy()

    # keep track of loss and accuracy
    train_acc_record = []
    val_acc_record = []
    train_loss_record = []
    val_loss_record = []

    iterMax_train = int(np.floor(M_train / minibatch_size))
    iterMax_val = int(np.floor(M_val / minibatch_size))

    # START TRAINING
    for t in range(num_epoch):
        print('At epoch ' + str(t + 1))
        if (t % period == 0) and (t != 0):
            alpha = alpha * 0.1

        # shuffle data order
        idx_order = np.random.permutation(M_train)

        # Training and validation accuracy and loss lists
        train_acc, train_loss = 0.0, 0.0
        train_count = 0
        val_acc, val_loss = 0.0, 0.0
        val_count = 0.0

        for i in range(iterMax_train):
            # get a minibatch of data
            subset_indices = idx_order[i * minibatch_size : (i + 1) * minibatch_size]
            x = Xtrain[subset_indices]
            y = Ytrain[subset_indices].reshape((5, 1))

            # forward calls
            a1 = neuralNet['L1'].forward(x)
            h1 = neuralNet['nonlinear1'].forward(a1)
            a2 = neuralNet['L2'].forward(h1)

            # loss
            loss = neuralNet['loss'].compute_loss(a2, y)

            # backward calls
            grad_a2 = neuralNet['loss'].compute_lossGrad(a2)
            grad_d1 = neuralNet['L2'].backward(h1, grad_a2)
            grad_a1 = neuralNet['nonlinear1'].backward(a1, grad_d1)

            grad_x = neuralNet['L1'].backward(x, grad_a1)

            # Update weights and biases
            neuralNet = miniBatchSGD(neuralNet, lambd, alpha)

        # compute TRAINING loss and accuracy
        for i in range(iterMax_train):
            subset_indices = np.arange(i * minibatch_size, (i + 1) * minibatch_size)
            x = Xtrain[subset_indices]
            y = Ytrain[subset_indices].reshape((5, 1))

            # forward calls
            a1 = neuralNet['L1'].forward(x)
            h1 = neuralNet['nonlinear1'].forward(a1)
            a2 = neuralNet['L2'].forward(h1)
            train_label_predictions = np.argmax(a2, axis=1).reshape((a2.shape[0], 1))

            # compute loss and accuracy
            loss = neuralNet['loss'].compute_loss(a2, y)
            train_loss += loss
            train_acc += np.sum(train_label_predictions == y)
            train_count += len(y)

        # training loss accuracy
        train_loss = train_loss
        train_acc = train_acc / train_count
        train_acc_record.append(train_acc)
        train_loss_record.append(train_loss)

        print('Training loss at epoch ' + str(t + 1) + ' is ' + str(train_loss))
        print('Training accuracy at epoch ' + str(t + 1) + ' is ' + str(train_acc))

        # accuracy over validation data
        for i in range(iterMax_val):
            subset_indices = np.arange(i * minibatch_size, (i + 1) * minibatch_size)
            x = Xtrain[subset_indices]
            y = Ytrain[subset_indices].reshape((5,1))

            # forward calls
            a1 = neuralNet['L1'].forward(x)
            h1 = neuralNet['nonlinear1'].forward(a1)
            a2 = neuralNet['L2'].forward(h1)
            val_label_predictions = np.argmax(a2, axis=1).reshape((a2.shape[0], 1))

            # validation loss and accuracy
            loss = neuralNet['loss'].compute_loss(a2, y)
            val_loss += loss
            val_acc += np.sum(val_label_predictions == y)
            val_count += len(y)

        val_loss_record.append(val_loss)
        val_acc = val_acc / val_count
        val_acc_record.append(val_acc)

        print('Validation accuracy at epoch ' + str(t + 1) + ' is ' + str(val_acc))

    print('Training finished!')

    # Load test data
    #-------------------
    x_test = np.genfromtxt('test_image.csv', delimiter=',')
    x_test = x_test/255

    # predict using trained neural net
    a1 = neuralNet['L1'].forward(x_test)
    h1 = neuralNet['nonlinear1'].forward(a1)
    a2 = neuralNet['L2'].forward(h1)
    test_label_predictions = np.argmax(a2, axis=1).reshape((a2.shape[0], 1))

    # Compute test error
    y_test = np.genfromtxt('test_label.csv', delimiter=',')
    y_test = y_test.reshape((y_test.shape[0], 1))
    test_acc = np.sum(test_label_predictions == y_test)
    test_acc = test_acc/y_test.shape[0]
    print("Test accuracy: ", test_acc)

    write_output(test_label_predictions)
    return 1


def write_output(predicted_labels):
    """
    function to write output to file.
    :param predicted_labels: label predictions for test dataset
    """
    np.savetxt("test_predictions.csv", predicted_labels, delimiter=",", fmt="%d")


if __name__ == "__main__":
    tStart = time.time()
    if len(sys.argv)>1:
        train_file_filename = sys.argv[1]
        train_label_filename = sys.argv[2]
        test_image_filename = sys.argv[3]

    nn_params = {"nodes_l1": 1000, "nodes_l2": 10, "alpha": 0.01, "lambda": 0.0, "num_epoch": 10, "minibatch_size": 5}
    dummy = trainNetwork(nn_params)

    print(time.time() - tStart)
