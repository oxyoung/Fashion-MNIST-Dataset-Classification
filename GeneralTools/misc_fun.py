"""
This file contains definition for FLAGS.
"""
import tensorflow as tf
import os
flags = tf.flags
FLAGS = tf.flags.FLAGS

# This is a bug of jupyter notebook
flags.DEFINE_string('f', '', "Empty flag to suppress UnrecognizedFlagError: Unknown command line flag 'f'.")

# local machine configuration
flags.DEFINE_integer('NUM_GPUS', 1, 'Number of GPUs in the local system.')
flags.DEFINE_float('EPSI', 1e-8, 'The smallest positive number to consider.')

# library info
flags.DEFINE_string('TENSORFLOW_VERSION', '1.13.1', 'Version of TensorFlow for the current project.')
flags.DEFINE_string('CUDA_VERSION', '9.0', 'Version of CUDA for the current project.')
flags.DEFINE_string('CUDNN_VERSION', '7.3.1', 'Version of CuDNN for the current project.')

# working directory info
flags.DEFINE_string('DEFAULT_IN', 'C:/Users/oxyoung/Desktop/Project-1-AI-for-Mechatronics-MCEN90048--master/data/fashion', 'Default input folder.')
flags.DEFINE_string('DEFAULT_OUT', 'C:/Users/oxyoung/Desktop/Project-1-AI-for-Mechatronics-MCEN90048--master/Results/FashionMNIST', 'Default output folder.')
flags.DEFINE_string(
    'DEFAULT_DOWNLOAD', 'C:/Users/oxyoung/Desktop/Project-1-AI-for-Mechatronics-MCEN90048--master/data/fashion',
    'Default folder for downloading large datasets.')
flags.DEFINE_string(
    'SUMMARY_FOLDER', os.path.join(FLAGS.DEFAULT_OUT, 'MLPBaseline', 'Adam_event_01'),
    'Default folder for summary files.')
flags.DEFINE_string(
    'TIMELINE_FOLDER', os.path.join(FLAGS.DEFAULT_OUT, 'MLPBaseline', 'Adam_timelines'),
    'Default folder for timeline files.')
flags.DEFINE_string(
    'LOG_FOLDER', os.path.join(FLAGS.DEFAULT_OUT, 'MLPBaseline', 'Adam_train_log'),
    'Default folder for training log files.')
flags.DEFINE_string(
    'CHECKINGPOINT_FOLDER', os.path.join(FLAGS.DEFAULT_OUT, 'MLPBaseline', 'Adam_ckpt_01'),
    'Default folder for checkpoint files.')
flags.DEFINE_string(
    'VISUALIZATION_FOLDER', os.path.join(FLAGS.DEFAULT_OUT, 'DataVisual'),
    'Default folder for visualization files.')
flags.DEFINE_string(
    'EXCEL_FOLDER', 'C:/Users/oxyoung/Desktop/Project-1-AI-for-Mechatronics-MCEN90048--master/Plots/traning_data_excel/',
    'Default folder for Excel Data.')

# model hyper-parameters
# Optimizer options: GradientDescent, Adam, Momentum
flags.DEFINE_string('OPTIMIZER', 'Adam', 'The default gradient descent optimizer.')
# Regularization method options: batch, drop out, None
flags.DEFINE_string('REGULARIZATION', 'drop out', 'The default regularization method.')
flags.DEFINE_bool('VERBOSE', True, 'Define whether to print more info during training and test.')
