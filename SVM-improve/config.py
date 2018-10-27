import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_float('min_velocity', -1, 'The minimum speed of particle update')
flags.DEFINE_float('max_velocity', 1, 'The maximum speed of particle update')
flags.DEFINE_float('C', 1.0, 'C')
flags.DEFINE_float('train_percent', 1.0, 'percent of the trainData')
flags.DEFINE_float('test_percent', 0.8, 'percent of the testData')
flags.DEFINE_integer('max_epoch', 10, 'The maximum number of iterations')
flags.DEFINE_integer('n_particles', 20, 'number of particles')

########################Adaboost parameters#########################
flags.DEFINE_float('learning_rate', 0.1, 'learning rate of adaboost')
flags.DEFINE_integer('n_estimators', 2000, 'n_estimators')
flags.DEFINE_integer('max_depth', 4, 'max_depth')
flags.DEFINE_integer('min_samples_split', 2, 'min_samples_split')


cfg = tf.app.flags.FLAGS