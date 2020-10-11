#Jeffrey Bradley
#9/11/2020


# import statements
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import tensorflow as tf

#10 classes (of clothing) -> 0-9
#784 features
#weight matrix so that each feature is a percentage of each class

# Load in dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#print(x_train, y_train)

print(y_train.shape)
print(x_train.shape)

x_train = x_train/255 # normalize dataset
#y_train = y_train/255

x_test = x_test/255
#y_test = y_test/255

# split data into 80% training 20% testing 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20)
##
##
##


x_train = tf.reshape(x_train,[-1,784])# reshape train input to be a 1-D vector
#tf.reshape(y_train,[-1,784])

x_test = tf.reshape(x_test,[-1,784])
#tf.reshape(y_test,[-1,784]) # reshape test input to be a 1-D vector

#print(x_train.shape)

 
##
##
#dtype = tf.float64

#weight matrix should be 784x10
#bias is different, on a class level,
##one big component to every class so 10 biases
##shape (10,)

w0 = tf.Variable(tf.random.normal([784,10], dtype = tf.float64)) # Randomly initialize weight matrix 
bias = tf.Variable(tf.random.normal([10,], dtype = tf.float64)) # Randomly initialize bias vector

#print(w0.shape)

#print(type(bias))

      
##
##
def logistic_regression(x_values):
   log_r = tf.matmul(x_values,w0) + bias
   #log_r = tf.add(tf.matmul(x_values,w0), bias) # Matrix multiply x values with weight matrix and add bias
   return log_r
##
def cross_entropy(y_values, y_pred):
   #predictions come from logistic regression
   
   y_values = tf.one_hot(y_values,10) # One-hot encoding vector of y values
   #loss = tf.reduce_mean(tf.square(y_values - v_pred)) # calculate the loss
   loss = tf.nn.softmax_cross_entropy_with_logits(y_values, y_pred) #measures cross entropy between y-predictions and y trues

   #type of loss function
   #true y's are one hot encoded
   #y-pred are probabilities they belong to certain classes
   #labels -> true values
   #logits -> probability prediction applying logit function to outcome of yhat,so only log_r output
   
   return tf.reduce_mean(loss) # return the average loss
##
def accuracy(y_values, y_pred):
   #argmax takes the propability and turns them back to real values and says is the
   #max probability class the correct class
   # values have to be casted as a 32 bit integer
   y_values = tf.cast(y_values, dtype=tf.int32)
   predictions = tf.cast(tf.argmax(y_pred, axis=1),dtype=tf.int32)
   
   predictions = tf.equal(y_values,predictions) # how many y values equal their predictions
   return  tf.reduce_mean(tf.cast(predictions,dtype=tf.float32)) # return the average correct predictions
##
def gradient(x_values, y_values):
   with tf.GradientTape() as t: # initialize your gradient tape
      #yhat = x_values * w0 + bias# obtain your prediction values
      yhat = logistic_regression(x_values)
      loss  = cross_entropy(y_values, yhat) # calculate your loss
   return t.gradient(loss,[w0,bias]) # return the gradient calculation with the losses and parameter list
##
batches = 10000
learning_rate = 0.01
batch_size = 128

##
###slicing and shuffling the batches so that our model is not bias to the data
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat().shuffle(x_train.shape[0]).batch(batch_size)
##
optimizer1 = tf.optimizers.SGD(learning_rate) # use stochastic gradient descent optimizer
##
for batch_number, (batch_x, batch_y) in enumerate(dataset.take(batches), 1):
   gradient1 = gradient(batch_x, batch_y) # find the gradient using your function
   optimizer1.apply_gradients(zip(gradient1,[w0,bias])) # apply the gradients to your parameters, use zip to pair gradient with parameters

   yhat = logistic_regression(batch_x) # obtain predictions for your logistic regression
   loss = cross_entropy(batch_y, yhat) # calculate the loss
   accuracy1 = accuracy(batch_y, yhat) # calculate the accuracy
   print("Batch number: %i, loss: %f, accuracy: %f" % (batch_number, loss, accuracy1))
##
