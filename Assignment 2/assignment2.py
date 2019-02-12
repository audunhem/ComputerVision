import numpy as np
import matplotlib.pyplot as plt
import mnist
import tqdm

#mnist.init() #uncomment if files are not already downloaded

###############################################################################
#FUNCTION DECLARATIONS

def should_early_stop(validation_loss, num_steps=4):
    if len(validation_loss) < num_steps+1:
        return False

    is_increasing = [validation_loss[i] <= validation_loss[i+1] for i in range(-num_steps-1, -1)]
    return sum(is_increasing) == len(is_increasing)

def train_val_split(X, Y, val_percentage):
  """
    Selects samples from the dataset randomly to be in the validation set. Also, shuffles the train set.
    --
    X: [N, num_features] numpy vector,
    Y: [N, 1] numpy vector
    val_percentage: amount of data to put in validation set
  """
  dataset_size = X.shape[0]
  idx = np.arange(0, dataset_size)
  np.random.shuffle(idx)

  train_size = int(dataset_size*(1-val_percentage))
  idx_train = idx[:train_size]
  idx_val = idx[train_size:]
  X_train, Y_train = X[idx_train], Y[idx_train]
  X_val, Y_val = X[idx_val], Y[idx_val]
  return X_train, Y_train, X_val, Y_val

def onehot_encode(Y, n_classes=10):
    onehot = np.zeros((Y.shape[0], n_classes))
    onehot[np.arange(0, Y.shape[0]), Y] = 1
    return onehot

def bias_trick(X):
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)

def check_gradient_hidden(X, targets, w_ji, w_kj, epsilon, computed_gradient):
    """
        Checks if the gradient for the weights from the input layer to the hidden layer
        are computed correctly.
    """
    print("Checking gradient...")
    dw = np.zeros_like(w_ji)

    for k in range(w_ji.shape[0]):
        for j in range(w_ji.shape[1]):
            new_weight1, new_weight2 = np.copy(w_ji), np.copy(w_ji)
            new_weight1[k,j] += epsilon
            new_weight2[k,j] -= epsilon
            [_,output1] = feedforward(X, [new_weight1, w_kj])
            [_,output2] = feedforward(X, [new_weight2, w_kj])
            loss1 = cross_entropy_loss(output1, targets)
            loss2 = cross_entropy_loss(output2, targets)
            dw[k,j] = (loss1 - loss2) / (2*epsilon)

    maximum_aboslute_difference = abs(computed_gradient-dw).max()
    assert maximum_aboslute_difference <= epsilon**2, "Absolute error was: {}".format(maximum_aboslute_difference)

def check_gradient_output(X, targets, w_kj, epsilon, computed_gradient):
    """
        Checks if the gradient for the weights from the hidden layer to the
        output layer are computed correctly.
    """
    print("Checking gradient...")
    dw = np.zeros_like(w_kj)
    for k in range(w_kj.shape[0]):
        for j in range(w_kj.shape[1]):
            new_weight1, new_weight2 = np.copy(w_kj), np.copy(w_kj)
            new_weight1[k,j] += epsilon
            new_weight2[k,j] -= epsilon
            output1 = forward_output(X, new_weight1)
            output2 = forward_output(X, new_weight2)
            loss1 = cross_entropy_loss(output1, targets)
            loss2 = cross_entropy_loss(output2, targets)
            dw[k,j] = (loss1 - loss2) / (2*epsilon)
    maximum_aboslute_difference = abs(computed_gradient-dw).max()
    assert maximum_aboslute_difference <= epsilon**2, "Absolute error was: {}".format(maximum_aboslute_difference)

def softmax(a):
    a_exp = np.exp(a)
    return a_exp / a_exp.sum(axis=1, keepdims=True)

def sigmoid(a):
    return 1/(1 + np.exp(-a))

def improved_sigmoid(a):
    return 1.7159*np.tanh(2*a/3)

def ReLU(a):
    return np.where(a > 0, a,0)

def dReLU(X,w):
    dx = np.ones_like(np.dot(X,w.T))
    dx[np.dot(X,w.T) < 0] = 0
    return dx

def LeakyReLU(a):
    return np.where(a > 0, a, 0.01*a)

def dLeakyReLU(X,w, alpha =0.01):
    #return np.where(np.dot(X,w.T) > 0, 1., 0.01)
    dx = np.ones_like(np.dot(X,w.T))
    dx[np.dot(X,w.T) < 0] = alpha
    return dx

def ELU(a, alpha = 0.005):
    return np.where(a > 0, a, alpha*(np.exp(a)-1))

def dELU(X,w, alpha =0.005):
    dx = np.ones_like(np.dot(X,w.T))
    under_zero = np.dot(X,w.T) < 0
    dx = np.add(under_zero.astype(int)*alpha*np.exp(np.dot(X,w.T)),dx)
    return dx

def calculate_derivative(A, weights, index):
    if (activation_function is 'improved_sigmoid'):
        return 1.7159*2/3*(1 - np.tanh(2*np.dot(A[-(index+1)],weights[-(index+1)].T)/3)**2).T
    elif (activation_function is 'sigmoid'):
        return (A[-index]*(1-A[-index])).T
    elif (activation_function is 'ReLU'):
        return dReLU(A[-(index+1)],weights[-(index+1)])
    elif (activation_function is 'leakyReLU'):
        return dLeakyReLU(A[-(index+1)],weights[-(index+1)])
    return dELU(A[-(index+1)],weights[-(index+1)])

def forward_hidden(X, w):
    a = X.dot(w.T)
    if (activation_function is 'improved_sigmoid'):
        return improved_sigmoid(a)
    elif (activation_function is 'sigmoid'):
        return sigmoid(a)
    elif (activation_function is 'ReLU'):
        return ReLU(a)
    elif (activation_function is 'leakyReLU'):
        return LeakyReLU(a)
    return ELU(a)

def forward_output(X, w):
    a = X.dot(w.T)
    return softmax(a)

def feedforward(X, weights):
    A = []
    A.append(forward_hidden(X,weights[0]))
    for i in range(1,len(weights)-1):
        A.append(forward_hidden(A[i-1],weights[i]))
    output = forward_output(A[-1], weights[-1])
    return A, output

def calculate_accuracy(output, targets):
    predictions = output.argmax(axis=1)
    targets = targets.argmax(axis=1)
    return (predictions == targets).mean()

def cross_entropy_loss(output, targets):
    assert output.shape == targets.shape
    log_y = np.log(output)
    cross_entropy = -targets * log_y
    return cross_entropy.mean() #equivalent to summing and dividing by batch_size * num_classes

def gradient_descent_output_layer(A, outputs, targets, weights, momentum, dropouts, learning_rate, should_check_gradient):
    """
        Preforms gradient descent for the weights between the hidden layer and
        the output layer.
    """
    w_down = weights[-1]
    normalization_factor = A[-1].shape[0] * targets.shape[1] # batch_size * num_classes
    delta_k = -(targets - outputs).T
    dw = delta_k.dot(A[-1])
    dw = dw / normalization_factor # Normalize gradient equally as loss normalization
    assert dw.shape == w_down.shape, "dw shape was: {}. Expected: {}".format(dw.shape, w.shape)

    if should_check_gradient:
        check_gradient_output(A[-1], targets, w_down, 1e-2,  dw)

    if nesterov_momentum:
        v_previous = momentum[-1]
        v_new = mu*v_previous - learning_rate * dw
        w_down = w_down + v_new
        return w_down, v_new
    w_down = w_down - learning_rate * dw
    return w_down


def gradient_descent_hidden_layer(index, A, outputs, targets, weights, momentum, dropouts, learning_rate, should_check_gradient):
    normalization_factor = A[-1].shape[0] * targets.shape[1] # batch_size * num_classes
    w_down = weights[index]
    w_up = weights[index+1]
    weights_momentum = np.add(weights, momentum)
    delta = -(targets - outputs).T
    for i in range(1,len(weights)-index):
        derivative = (calculate_derivative(A, weights_momentum, i)*dropouts[-i]).T
        delta = derivative*np.dot(weights_momentum[-i].T,delta)
    dw = delta.dot(A[index])
    dw = dw / normalization_factor # Normalize gradient equally as loss normalization
    assert dw.shape == w_down.shape, "dw shape was: {}. Expected: {}".format(dw.shape, w_down.shape)
    if should_check_gradient:
        check_gradient_hidden(A[index], targets, w_down, w_up, 1e-2,  dw)

    if nesterov_momentum:
        v_previous = momentum[index]
        v_new = mu*v_previous - learning_rate * dw
        w_down = w_down + v_new
        return w_down, v_new
    w_down = w_down - learning_rate * dw
    return w_down

def backpropagate(X, targets, weights, momentum, learning_rate, should_check_gradient):
    new_weights = []
    new_momentum = []
    dropouts = []
    if nesterov_momentum:
        [A, outputs] = feedforward(X, np.add(weights, momentum))
    [A, outputs] = feedforward(X, weights)
    A.insert(0,X)
    for i in range(len(A)):
        dropouts.append(dropout(A[i]))
        A[i] = A[i]*dropouts[i]

    if nesterov_momentum:

        [new_weight, v_new] = gradient_descent_output_layer(A, outputs, targets, weights, momentum, dropouts, learning_rate, should_check_gradient)
        new_momentum.insert(0,v_new)
        new_weights.insert(0,new_weight)
    else:
        new_weight = gradient_descent_output_layer(A, outputs, targets, weights, momentum, dropouts, learning_rate, should_check_gradient)
        new_weights.insert(0,new_weight)

    for i in range(len(weights)-2,-1,-1):
        if nesterov_momentum:
            [new_weight, v_new] = gradient_descent_hidden_layer(i, A, outputs, targets, weights, momentum, dropouts, learning_rate, should_check_gradient)
            new_momentum.insert(0,v_new)
            new_weights.insert(0,new_weight)
        else:
            new_weight = gradient_descent_hidden_layer(i, A, outputs, targets, weights, momentum, dropouts ,learning_rate, should_check_gradient)
            new_weights.insert(0,new_weight)
    if nesterov_momentum:
        return new_weights, new_momentum
    return new_weights, momentum

def weight_initialization(input_units, output_units):
    weight_shape = (output_units, input_units)
    if (normal_distributed_weights):
        mean = 0
        variance = 1/np.sqrt(input_units)
        return np.random.normal(mean, variance, weight_shape)
    return np.random.uniform(-1, 1, weight_shape)

def shuffle_train_set(X_train, Y_train):
    All_indexes = np.arange(Y_train.shape[0])
    np.random.shuffle(All_indexes)
    Train_indexes = All_indexes[:(Y_train.size)]
    X_train_new = X_train[Train_indexes]
    Y_train_new = Y_train[Train_indexes]
    return X_train_new, Y_train_new

def bit_shift_image_UP(X_for_augment):
    for img_num in range(X_for_augment.shape[0]):
        for j in range(HEIGHT):
            for i in range(WIDTH):
                if(j+num_bits_shifted<HEIGHT):
                    X_for_augment[img_num][j][i] = X_for_augment[img_num][j+num_bits_shifted][i]
                else:
                    X_for_augment[img_num][j][i] = -1.
    return X_for_augment

def bit_shift_image_DOWN(X_for_augment):
    for img_num in range(X_for_augment.shape[0]):
        for j in range(HEIGHT):
            for i in range(WIDTH):
                if(HEIGHT-(j+num_bits_shifted)>0):
                    X_for_augment[img_num][HEIGHT-1-j][i] = X_for_augment[img_num][HEIGHT-1-(j+num_bits_shifted)][i]
                else:
                    X_for_augment[img_num][HEIGHT-1-j][i] = -1.
    return X_for_augment

def bit_shift_image_LEFT(X_for_augment):
    for img_num in range(X_for_augment.shape[0]):
        for j in range(HEIGHT):
            for i in range(WIDTH):
                if(i+num_bits_shifted<WIDTH):
                    X_for_augment[img_num][j][i] = X_for_augment[img_num][j][i+num_bits_shifted]
                else:
                    X_for_augment[img_num][j][i] = -1.
    return X_for_augment

def bit_shift_image_RIGHT(X_for_augment):
    for img_num in range(X_for_augment.shape[0]):
        for j in range(HEIGHT):
            for i in range(WIDTH):
                if(WIDTH-(i+num_bits_shifted)>0):
                    X_for_augment[img_num][j][WIDTH-1-i] = X_for_augment[img_num][j][WIDTH-1-(i+num_bits_shifted)]
                else:
                    X_for_augment[img_num][j][WIDTH-1-i] = -1.
    return X_for_augment

def bit_shift_direction(X,random, direction):
    X= X[:,:-1] # Remove bias
    X = X.reshape(X.shape[0],28, 28)
    if(random):
        direction = np.random.randint(4)
    if(direction==0):
        X=bit_shift_image_UP(X)
    elif(direction==1):
        X=bit_shift_image_DOWN(X)
    elif(direction==2):
        X=bit_shift_image_LEFT(X)
    elif(direction==3):
        X=bit_shift_image_RIGHT(X)

    X = X.reshape(X.shape[0],28*28)
    X = bias_trick(X)
    return X

def dropout(data):
    dropout_vector = np.random.binomial(1, (1-dropout_probability), size=data.shape)/(1-dropout_probability)
    return dropout_vector


###############################################################################
#DATA INITIALIZATION

X_train, Y_train, X_test, Y_test = mnist.load()

X_train, X_test = (X_train / 127.5)-1, (X_test / 127.5)-1
X_train = bias_trick(X_train)
X_test = bias_trick(X_test)
Y_train, Y_test = onehot_encode(Y_train), onehot_encode(Y_test)

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)


###############################################################################
#FEATURES

activation_function = 'ReLU' #can use: sigmoid, improved_sigmoid, ELU, ReLU, LeakyReLU
shuffle_training_examples = True
normal_distributed_weights = True
nesterov_momentum = True

#network topology
hidden_layer_units = [64, 64]

#bonus
dropout_active = True
pixel_shifting = False


###############################################################################
#VARIABLE DECLARATIONS

# Hyperparameters

batch_size = 128
learning_rate = 0.5
num_batches = X_train.shape[0] // batch_size
should_check_gradient = False
check_step = num_batches // 10
max_epochs = 5
v_1 = 0
v_2 = 0
v_output = 0
mu = 0

WIDTH = 28
HEIGHT = 28
num_bits_shifted = 0
dropout_probability = 0

if dropout_active:
    dropout_probability = 0.2

if nesterov_momentum:
    mu = 0.5

activate_bias = True

# Tracking variables
TRAIN_LOSS = []
TEST_LOSS = []
VAL_LOSS = []
TRAIN_ACC = []
TEST_ACC = []
VAL_ACC = []
PREVIOUS_WEIGHTS = [0,0,0,0]


###############################################################################
#TRAINING

def train_loop(X_train, Y_train, should_check_gradient):
    print('Starting...')

    weights = []
    weights.append(weight_initialization(X_train.shape[1],hidden_layer_units[0]))
    for i in range(len(hidden_layer_units)-1):
        weights.append(weight_initialization(hidden_layer_units[i],hidden_layer_units[i+1]))
    weights.append(weight_initialization(hidden_layer_units[-1],Y_train.shape[1]))
    weights = np.array(weights)
    momentum = [0]*(len(hidden_layer_units)+1)

    for e in range(max_epochs): # Epochs
        if (shuffle_training_examples):
            X_train, Y_train = shuffle_train_set(X_train, Y_train)

        learning_rate_annealing = learning_rate/(1+e/max_epochs) #use if needed

        for i in tqdm.trange(num_batches):
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            Y_batch = Y_train[i*batch_size:(i+1)*batch_size]

            if pixel_shifting:
                X_batch = bit_shift_direction(X_batch,True,0)

            #Backpropagation
            [weights, momentum] = backpropagate(X_batch, Y_batch, weights, momentum, learning_rate, should_check_gradient)

            activate_bias = False
            should_check_gradient = False #checking gradient only for the first batch in the first epoch
            if i % check_step == 0:
                # Loss
                [_,train_output] = feedforward(X_train, weights)
                [_,test_output] = feedforward(X_test, weights)
                [_,val_output] = feedforward(X_val, weights)
                TRAIN_LOSS.append(cross_entropy_loss(train_output, Y_train))
                TEST_LOSS.append(cross_entropy_loss(test_output, Y_test))
                VAL_LOSS.append(cross_entropy_loss(val_output, Y_val))


                TRAIN_ACC.append(calculate_accuracy(train_output, Y_train))
                TEST_ACC.append(calculate_accuracy(test_output, Y_test))
                VAL_ACC.append(calculate_accuracy(val_output, Y_val))

                PREVIOUS_WEIGHTS.append([weights])
                PREVIOUS_WEIGHTS.pop(0)

                if should_early_stop(VAL_LOSS):
                    print(VAL_LOSS[-4:])
                    print("early stopping.")
                    ideal_weights = PREVIOUS_WEIGHTS[0]
                    return ideal_weights
    return weights

final_weights = train_loop(X_train, Y_train, should_check_gradient)


###############################################################################
#PLOTTING

print(VAL_LOSS[-4:])
print(VAL_ACC[-10:])
plt.plot(TRAIN_LOSS, label="Training loss")
plt.plot(TEST_LOSS, label="Testing loss")
plt.plot(VAL_LOSS, label="Validation loss")
plt.legend()
plt.ylim([0, 1])
plt.show()

plt.clf()
plt.plot(TRAIN_ACC, label="Training accuracy")
plt.plot(TEST_ACC, label="Testing accuracy")
plt.plot(VAL_ACC, label="Validation accuracy")
plt.ylim([0, 1.0])
plt.legend()
plt.show()
