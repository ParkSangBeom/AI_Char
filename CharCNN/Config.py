import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('name', "CHAR_CNN", 'Project Name')
flags.DEFINE_string('train_data_path', "./Data/ratings_train.txt", 'Training Data Path')
flags.DEFINE_string('test_data_path', "./Data/ratings_test.txt", 'Training Data Path')
