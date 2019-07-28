import tensorflow as tf
import time
import imageio
import numpy as np
import os
import warnings
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.client import timeline
from GeneralTools.misc_fun import FLAGS
from GeneralTools.my_input import InputPipeline

class TrainingModel(object):
    """
    This class define the MPL model including training and testing methods
    """
    def __init__(self):
        """ Initialize the model
        """
        self.fashion_data = input_data.read_data_sets(FLAGS.DEFAULT_IN, one_hot=True)

    def add_layer(self, inputs, input_size, output_size, layer_name, activation_function=None,):
        """ Create one layer and return the outputs of the layer
        :param inputs: inputs of the layer
        :param input_size: input size of the layer, must be an integer
        :param output_size: output size of the layer, must be an integer
        :param activation_function: the activation function of the layer, default is not to use any function.
        """
        init_bias = 0
        with tf.variable_scope(layer_name):
            with tf.name_scope("weights"):
                weights = tf.Variable(tf.random_normal_initializer(stddev=0.01)([input_size, output_size]),
                                      dtype=tf.float32, name='weights')
                tf.summary.histogram(layer_name + "_weights", weights)
            with tf.name_scope("biases"):
                biases = tf.Variable(tf.zeros([1, output_size])+init_bias, dtype=tf.float32, name='biases')
                tf.summary.histogram(layer_name + "_biases", biases)
            with tf.name_scope("weight_plus_bias"):
                weight_plus_bias = tf.matmul(inputs, weights) + biases

                if FLAGS.REGULARIZATION is 'batch':
                    weight_plus_bias = self.batch_normalization(weight_plus_bias, output_size)

            outputs = activation_function(weight_plus_bias)
        return outputs

    def create_placeholders(self, input_size, output_size):
        """ This function creates a placeholder for tensor
        :param input_size: size of the input, must be integer
        :param output_size: size of the output, must be integer
        """
        x_sample = tf.placeholder(tf.float32, [None, input_size], name="x_input")  # 28x28
        y_sample = tf.placeholder(tf.float32, [None, output_size], name="y_input")
        return x_sample, y_sample

    def batch_normalization(self, input, output_size):
        """ This function defines batch normalization method
        :param input: input data for batch normalization.
        :param output_size: size of the output, must be integer
        """
        # Batch Normalization
        mean, var = tf.nn.moments(input, axes=[0])
        scale = tf.Variable(tf.ones([output_size]))
        shift = tf.Variable(tf.zeros([output_size]))
        epsilon = 0.001

        # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.4)
        def mean_var_with_update():
            ema_apply_op = ema.apply([mean, var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(mean), tf.identity(var)

        mean, var = mean_var_with_update()
        output = tf.nn.batch_normalization(input, mean, var, shift, scale, epsilon)
        return output

    def train(self, train_data, train_label, valid_data=None, valid_label=None, learning_rate=0.01,
              max_epochs=1000, keep_training=False):
        """ This function defines the training process of the model
        :param train_data: the input training data, must be a shape like [sample_number, data]
        :param train_label: the input label of the training data, must be a shape like [sample_number, labels]
        :param valid_data: the give validation data, must be a shape like [sample_number, data]
        :param valid_label: the give validation labels, must be a shape like [sample_number, labels]
        :param learning_rate: the learning rate of the optimizer
        :param max_epochs: maximum epoch of the training
        :param keep_training: determine to continue train the model
        """

        if valid_data or valid_label is None:
            valid_data = self.fashion_data.test.images
            valid_label = self.fashion_data.test.labels
        # Initialize all parameters
        # Fashion data images size
        pixel_size = train_data.shape[1]
        class_number = train_label.shape[1]
        graph = tf.Graph()
        with graph.as_default(), tf.device('cpu:0'):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # Define input sample for each layers
            with tf.name_scope('Inputs'):
                x_sample, y_sample = self.create_placeholders(pixel_size, class_number)
            # Define output layer
            hidden_layer1 = self.add_layer(x_sample, pixel_size, 300, 'layer1', activation_function=tf.nn.sigmoid)
            if FLAGS.REGULARIZATION == 'drop out':
                keep_prob = tf.placeholder(tf.float32)
                probability = 0.5
                hidden_layer1 = tf.nn.dropout(hidden_layer1, keep_prob)
            prediction = self.add_layer(hidden_layer1, 300, class_number, 'layer2', activation_function=tf.nn.softmax)
            if FLAGS.REGULARIZATION == 'drop out':
                prediction = tf.nn.dropout(prediction, keep_prob)

            with tf.name_scope("Training"):
                # Define loss function using cross entropy
                with tf.name_scope("Loss_Function"):
                    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_sample *
                        tf.log(prediction + FLAGS.EPSI), reduction_indices=1))
                    tf.summary.scalar("Loss/train", cross_entropy)
                if FLAGS.OPTIMIZER == 'Adam':
                    training = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

                elif FLAGS.OPTIMIZER == 'Momentum':
                    training = tf.train.MomentumOptimizer(learning_rate, momentum=0.7).minimize(cross_entropy,
                                                                                  global_step=global_step)
                else:
                    training = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy,
                                                                                         global_step=global_step)
                # else:
                #     print('Please select a correct optimizer.')
                with tf.name_scope("Accuracy"):
                    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_sample, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
                    tf.summary.scalar('accuracy', accuracy)

            # Create Data Pipeline
            batch_size = 512
            minibatch = InputPipeline(batch_size, train_data, train_label)
            minibatch.schedule(buffer_size=10000)
            x_batch, y_batch = minibatch.next()
            iterator_initializer, mini_dict = minibatch.initializer()
            num_minibatch = int(train_data.shape[0] / batch_size)

            folder_name = (FLAGS.SUMMARY_FOLDER, FLAGS.TIMELINE_FOLDER, FLAGS.LOG_FOLDER, FLAGS.CHECKINGPOINT_FOLDER)
            for folder in folder_name:
                if not os.path.exists(folder):
                    os.makedirs(folder)
            f_result = open(FLAGS.LOG_FOLDER + '/training result_{:.0f}.txt'.format(time.time()), 'a')

            # Initialize a session and saver
            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()

            # Initialize Tensorboard Summary
            merged_train = tf.summary.merge_all()
            merged_test = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(FLAGS.SUMMARY_FOLDER + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(FLAGS.SUMMARY_FOLDER + '/test', sess.graph)

            # Set runtime statistics option
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            start_time = time.time()
            if FLAGS.VERBOSE:
                print('Training Start!')
                print('Optimizer:', FLAGS.OPTIMIZER)
                print('Regularization method:', FLAGS.REGULARIZATION)
            # Start training process and iteration is determined by max_epochs
            if keep_training is True:
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.CHECKINGPOINT_FOLDER))
                if FLAGS.VERBOSE:
                    print("Restore model from ", FLAGS.CHECKINGPOINT_FOLDER)
            for epoch in range(max_epochs):
                sess.run(iterator_initializer, mini_dict)
                # Start an epoch and training through all mini batches
                for step in range(num_minibatch):
                    global_step_value = sess.run(global_step)
                    batch_x_sample, batch_y_sample = sess.run([x_batch, y_batch])
                    if FLAGS.REGULARIZATION == 'drop out':
                        sess.run(training, feed_dict={x_sample: batch_x_sample,
                                                      y_sample: batch_y_sample, keep_prob: probability})
                    else:
                        sess.run(training, feed_dict={x_sample: batch_x_sample, y_sample: batch_y_sample})
                epoch = int(global_step_value / num_minibatch) + 1
                # Start learning process and writer summary every 30 epochs
                if epoch % 30 == 0 or epoch == max_epochs:
                    if FLAGS.REGULARIZATION == 'drop out':
                        loss_train = sess.run(cross_entropy, feed_dict={x_sample: train_data,
                                                                        y_sample: train_label, keep_prob: 1})
                        loss_test = sess.run(cross_entropy, feed_dict={x_sample: valid_data,
                                                                       y_sample: valid_label, keep_prob: 1})
                        acc_test = sess.run(accuracy, feed_dict={x_sample: valid_data,
                                                                 y_sample: valid_label, keep_prob: 1})
                        train_summary = sess.run(merged_train,
                                                 feed_dict={x_sample: train_data,
                                                            y_sample: train_label, keep_prob: 1},
                                                 options=run_options, run_metadata=run_metadata)
                        test_summary = sess.run(merged_test, feed_dict={x_sample: valid_data,
                                                                        y_sample: valid_label, keep_prob: 1},
                                                options=run_options, run_metadata=run_metadata)
                    else:
                        loss_train = sess.run(cross_entropy, feed_dict={x_sample: train_data, y_sample: train_label})
                        loss_test = sess.run(cross_entropy, feed_dict={x_sample: valid_data, y_sample: valid_label})
                        acc_test = sess.run(accuracy, feed_dict={x_sample: valid_data, y_sample: valid_label})
                        train_summary = sess.run(merged_train,
                                                 feed_dict={x_sample: train_data, y_sample: train_label},
                                                 options=run_options, run_metadata=run_metadata)
                        test_summary = sess.run(merged_test, feed_dict={x_sample: valid_data, y_sample: valid_label},
                                                options=run_options, run_metadata=run_metadata)

                    # Add tensorboard summary
                    train_writer.add_run_metadata(run_metadata, 'epoch%d' % epoch)
                    train_writer.add_summary(train_summary, global_step=epoch)
                    test_writer.add_summary(test_summary, global_step=epoch)
                    result_log = "Epoch: {}, Accuracy: {:.3f}, Loss train: {:.3f}," \
                                 " Loss test: {:.3f}\n".format(epoch, acc_test, loss_train, loss_test)
                    if FLAGS.VERBOSE:
                        print('Adding run metadata for epoch:', epoch)
                        print(result_log)
                    # Save training logs to txt file
                    f_result.write('Adding run metadata for epoch:{}\n'.format(epoch))
                    f_result.write(result_log)
                    # Save runtime statistic results
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open(FLAGS.TIMELINE_FOLDER + '/timeline_epoch_{}.json'.format(epoch), 'w') as f_tracing:
                        f_tracing.write(chrome_trace)
            duration = time.time() - start_time
            if FLAGS.VERBOSE:
                print('Training for {} epochs took {:.3f} sec.\n'.format(epoch, duration))
                print('Training process finished')
            f_result.write('Training for {} epochs took {:.3f} sec.'.format(epoch, duration))
            f_result.close()
            # Save trained model to .ckpt files
            saver.save(sess, FLAGS.CHECKINGPOINT_FOLDER + '/project1_trained_model')
            sess.close()

    def test_model(self, test_images, test_labels):
        """ This function tests the trained model. The trained model is restored from the checkpoint_folder
        defined below. Please make sure the checkpoint folder path in misc_fun.py is correct.
        :param test_images: the give testing images, must be a shape like [sample_number, images]
        :param test_labels: the give testing labels, must be a shape like [sample_number, labels]
        """
        pixel_size = 784
        class_number = 10
        x_sample, y_sample = self.create_placeholders(test_images.shape[1], test_labels.shape[1])
        #
        hidden_layer1 = self.add_layer(x_sample, pixel_size, 300, 'layer1', activation_function=tf.nn.sigmoid)
        prediction = self.add_layer(hidden_layer1, 300, class_number, 'layer2', activation_function=tf.nn.softmax)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_sample, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.CHECKINGPOINT_FOLDER))
            acc_test = sess.run(accuracy, feed_dict={x_sample: test_images, y_sample: test_labels})
            if FLAGS.VERBOSE:
                print('Accuracy of the test model is {:.3f}'.format(acc_test))
                print('Test process finished\n')

    def visualization(self):
        """ This function defines the visualization process. The folder saving visualization is determined by
        VISUALIZATION_FOLDER, which can be altered in misc_fun.py file.
        """
        fashion_data = input_data.read_data_sets(FLAGS.DEFAULT_IN, one_hot=True)
        graph = tf.Graph()
        with graph.as_default():
            batch_size = 400
            x1, yv = fashion_data.train.next_batch(batch_size)
            xv = np.array(list(x1)).reshape(400, 28, 28)

            # prepare folder
            filename = 'Fashionmnist'
            if not os.path.exists(FLAGS.VISUALIZATION_FOLDER):
                os.makedirs(FLAGS.VISUALIZATION_FOLDER)
            embedding_path = os.path.join(FLAGS.VISUALIZATION_FOLDER, filename + '_embedding.ckpt')
            sprite_path = os.path.join(FLAGS.VISUALIZATION_FOLDER, filename + '.png')
            label_path = os.path.join(FLAGS.VISUALIZATION_FOLDER, filename + '_label.tsv')

            # prepare data, sprite images and files
            embedding_data = np.reshape(xv, (batch_size, -1))
            images = xv  # shape [400, 28, 28]
            image_size = xv.shape[1:]  # [28, 28]
            labels = yv

            # write label to file
            if os.path.isfile(label_path):
                warnings.warn(
                    'Label file {} already exists, thus this step is ignored.'.format(label_path))
            else:
                metadata_file = open(label_path, 'w')
                metadata_file.write('Name\tClass\n')
                for index, label in enumerate(labels):
                    metadata_file.write('%06d\t%s\n' % (index, str(label)))
                metadata_file.close()

            # write images to sprite
            if os.path.isfile(sprite_path):
                warnings.warn(
                    'Sprite file {} already exists, thus this step is ignored.'.format(sprite_path))
            else:
                # extend image shapes to [batch size, height, width, 3]
                if len(images.shape) == 3:  # if dimension of image is 3, extend it to 4
                    images = np.tile(images[..., np.newaxis], (1, 1, 1, 3))
                    if FLAGS.VERBOSE:
                        print('Shape of images has been changed to {}'.format(images.shape))
                if images.shape[3] == 1:  # if last dimension is 1, extend it to 3
                    images = np.tile(images, (1, 1, 1, 3))
                    if FLAGS.VERBOSE:
                        print('Shape of images has been changed to {}'.format(images.shape))

                # invert images for mnist
                images = 1 - images

                # Tile the individual thumbnails into an image
                mesh_num = (20, 20)
                new_shape = mesh_num + images.shape[1:]  # (20, 20, 28, 28, 3)
                images = images.reshape(new_shape).transpose((0, 2, 1, 3, 4))
                images = images.reshape(
                    (mesh_num[0] * images.shape[1], mesh_num[1] * images.shape[3]) + images.shape[4:])
                images = (images * 255).astype(np.uint8)
                # save images to file
                imageio.imwrite(sprite_path, images)

            # write data to ckpt
            if os.path.isfile(embedding_path):
                warnings.warn(
                    'Embedding file {} already exists, thus this step is ignored.'.format(embedding_path))
            else:
                # register a session
                with tf.Session() as sess:
                    # prepare a embedding variable
                    # note this must be a variable, not a tensor/constant
                    embedding_var = tf.Variable(embedding_data, name='em_data')
                    sess.run(embedding_var.initializer)
                    # configure the embedding projector
                    config = projector.ProjectorConfig()
                    embedding = config.embeddings.add()
                    embedding.tensor_name = embedding_var.name
                    # add metadata (label) to embedding
                    if label_path is not None:
                        embedding.metadata_path = label_path
                    # add sprite image to embedding
                    if sprite_path is not None:
                        embedding.sprite.image_path = sprite_path
                        embedding.sprite.single_image_dim.extend(image_size)
                    # finalize embedding setting
                    embedding_writer = tf.summary.FileWriter(FLAGS.VISUALIZATION_FOLDER)
                    projector.visualize_embeddings(embedding_writer, config)
                    embedding_saver = tf.train.Saver([embedding_var], max_to_keep=1)
                    embedding_saver.save(sess, embedding_path)
            if FLAGS.VERBOSE:
                print("Visualization finished")
                print("Results have been saved in {}\n".format(FLAGS.VISUALIZATION_FOLDER))
