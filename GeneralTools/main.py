
from GeneralTools.my_model import TrainingModel
from tensorflow.examples.tutorials.mnist import input_data
from GeneralTools.misc_fun import FLAGS
# from GeneralTools.my_plot import plot_dif_learningrate
FLAGS.DEFAULT_IN = 'C:/Users/oxyoung/Desktop/Project1/data/fashion'  # add data folder
FLAGS.DEFAULT_OUT = 'C:/Users/oxyoung/Desktop/Project1/Results/FashionMNIST'  # add folder to save the final results

# configurate the model hyper-parameters
learning_rate = 0.001
max_epochs = 300
# instantiate a model main
model = TrainingModel()
fashion_data = input_data.read_data_sets(FLAGS.DEFAULT_IN, one_hot=True)

# train the model
model.train(fashion_data.train.images, fashion_data.train.labels, learning_rate=0.001, max_epochs=300, keep_training=False)

#model.train(fashion_data.train.images, fashion_data.train.labels, learning_rate=0.01, max_epochs=20, keep_training=True)

model.test_model(fashion_data.test.images, fashion_data.test.labels)
model.visualization()

# plot_dif_learningrate()

# indicate the code run successfully
print('Code is successfully executed')


