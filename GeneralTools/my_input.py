import tensorflow as tf
from GeneralTools.misc_fun import FLAGS
class InputPipeline(object):
    """
    This class defines an input data pipeline.
    The code is modified based on richardwth/ai_for_mechatronics/TensorFlow_basics_02_session_and_input.ipynb
    data pipeline section
    """
    def __init__(self, batch_size, x_data, y_data, dtype=tf.float32, name='DataPipeline'):
        """ Initialize a pipeline class.
        :param batch_size: batch size of the pipeline
        :param x_data: input data given to the pipeline, must be a shape like [sample_number, data]
        :param y_data: output data given to the pipeline, must be a shape like [sample_number, data]
        :param dtype: data typo of the input and output data
        :param name: the name of the data pipeline
        """
        self.name_scope = name
        self.dtype = dtype
        self.batch_size = batch_size
        with tf.name_scope(name=name):
            # Create placeholder and initialize the dataset
            x_sample = tf.placeholder(self.dtype, x_data.shape)
            y_sample = tf.placeholder(tf.int32, y_data.shape)
            self.feed_dict = {x_sample: x_data, y_sample: y_data}
            self.num_samples = x_data.shape[0]
            self.dataset = tf.data.Dataset.from_tensor_slices({"images": x_sample, "labels": y_sample})
            # Apply map function
            self.dataset = self.dataset.map(lambda d: (d['images'], d['labels']))

        self.iterator = None

    def schedule(self, num_epoch=1, skip_count=None, shuffle=True, buffer_size=10000):
        """ Define a schedule of the pipeline
        :param num_epoch: epoch of the pipeline, if it is None or -1 will repeat the dataset infinite times
        :param skip_count: the number of data are skipped
        :param shuffle: define shuffling data or not
        :param buffer_size: Shuffling buffer size
        """
        with tf.name_scope(self.name_scope):
            if skip_count is None:
                skip_count = self.num_samples % self.batch_size
            if skip_count > 0:
                self.dataset = self.dataset.skip(skip_count)
                if FLAGS.VERBOSE:
                    print('{}: Number of {} instances skipped.'.format(
                        self.name_scope, skip_count))
            # shuffle
            if shuffle:
                self.dataset = self.dataset.shuffle(buffer_size)
            # make batch
            self.dataset = self.dataset.batch(self.batch_size)
            # repeat datasets for num_epoch
            self.dataset = self.dataset.repeat(num_epoch)
            # initialize an iterator
            self.iterator = self.dataset.make_initializable_iterator()

        return self  # facilitate method cascading

    def next(self):
        """ Define a function to get next batch
        """
        if self.iterator is None:
            self.schedule(self.batch_size)
        return self.iterator.get_next()

    def initializer(self):
        """ Define a function to initialize the pipeline
        """
        assert self.iterator is not None, \
            '{}: Batch must be provided.'.format(self.name_scope)
        return self.iterator.initializer, self.feed_dict

