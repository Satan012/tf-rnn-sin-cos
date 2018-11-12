import os
import shutil
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('train_samples_num', 1000, 'number of points in the train dataset')
tf.app.flags.DEFINE_float('sample_gap', 0.01, 'the interval of sampling')
tf.app.flags.DEFINE_integer('layer_num', 1, 'number of lstm layer')
tf.app.flags.DEFINE_integer('test_samples_num', 10000, 'number of points in the test dataset')
tf.app.flags.DEFINE_integer('units_num', 128, 'number of hidden units of lstm')
tf.app.flags.DEFINE_integer('epoch', 50, 'epoch of training step')
tf.app.flags.DEFINE_integer('batch_size', 64, 'mini_batch size')
tf.app.flags.DEFINE_integer('max_len', 10, 'we will use ten point to predict the value of 11th')
tf.app.flags.DEFINE_enum('model_state', 'predict', ["train", "predict"], 'model state')
tf.app.flags.DEFINE_boolean('debugging', False, 'delete log or not')
tf.app.flags.DEFINE_float('lr', 0.01, 'learning rate')
tf.app.flags.DEFINE_enum('function', 'cos', ['sin', 'cos'], 'select sin funtion or cosin function')


class RNN(object):
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.max_len])
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None])
        self.global_step = tf.train.create_global_step()

        self.input = tf.expand_dims(input=self.x, axis=-1)  # rnn标准的输入维度为 [batch_size, seq_len, dim_size]

    def build_rnn(self):
        with tf.variable_scope('lstm_layer'):
            cells = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.BasicLSTMCell(FLAGS.units_num) for _ in range(FLAGS.layer_num)])

            outputs, final_states = tf.nn.dynamic_rnn(cell=cells, inputs=self.input, dtype=np.float32)

            self.outputs = outputs[:, -1]  # 最后一个time step的输出才是我们真正需要的，因为是预测第11个time step的值

        with tf.variable_scope('output_layer'):
            self.predicts = tf.contrib.layers.fully_connected(self.outputs, 1, activation_fn=None)
            self.predicts = tf.reshape(tensor=self.predicts, shape=[-1])

    def build_train_op(self):
        with tf.variable_scope('train_op_layer'):
            self.loss = tf.reduce_mean(tf.square(self.y_ - self.predicts))
            # self.loss = tf.losses.mean_squared_error(self.predicts, self.y_)  # 与上一行效果等效，上一行为详细计算写法，可替换
            tf.summary.scalar(name='loss', tensor=self.loss)  # 配合tensorboard动态监控loss值下降情况

            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
            self.train_op = optimizer.minimize(self.loss, self.global_step)

    def build_net(self):
        self.build_rnn()
        self.build_train_op()
        self.merged_summary = tf.summary.merge_all()


def generate_data(seq):
    x = []
    y = []
    for i in range(len(seq) - FLAGS.max_len):
        x.append(seq[i:i + FLAGS.max_len])
        y.append(seq[i + FLAGS.max_len])

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


# 获取一个batch_size大小的数据
def get_batches(X, y):
    batch_size = FLAGS.batch_size
    for i in range(0, len(X), batch_size):
        begin_i = i
        end_i = i + batch_size if (i + batch_size) < len(X) else len(X)

        yield X[begin_i:end_i], y[begin_i:end_i]


def average_mse(real, predict):
    predict = np.array(predict)
    mse = np.mean(np.square(real - predict))
    return mse


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.function == 'sin':
        func = lambda x: np.sin(x)
        log_dir = 'log/sin/'
    else:
        func = lambda x: np.cos(x)
        log_dir = 'log/cos/'
    test_start = FLAGS.train_samples_num * FLAGS.sample_gap
    test_end = (FLAGS.train_samples_num + FLAGS.test_samples_num) * FLAGS.sample_gap
    train_x, train_y = generate_data(func(np.linspace(0, test_start, FLAGS.train_samples_num, dtype=np.float32)))
    tf.logging.info(
        'train dataset has been prepared. train_x shape: {}; train_y shape: {}'.format(train_x.shape, train_y.shape))

    test_x, test_y = generate_data(np.sin(np.linspace(test_start, test_end, FLAGS.test_samples_num, dtype=np.float32)))
    tf.logging.info(
        'test dataset has been prepared. test_x shape: {}; test_y shape: {}'.format(test_x.shape, test_y.shape))

    rnn_model = RNN()
    rnn_model.build_net()

    if FLAGS.debugging:
        if os.path.exists(log_dir):
            print('remove：' + log_dir)
            shutil.rmtree(log_dir)

    if FLAGS.model_state == 'train':
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    saver = tf.train.Saver()
    sv = tf.train.Supervisor(logdir=log_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=None,  # save summaries for tensorboard every 60 secs
                             save_model_secs=60,  # checkpoint every 60 secs
                             global_step=rnn_model.global_step)

    tf.logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session()
    tf.logging.info("Created session.")
    minLoss = 1000
    with sess_context_manager as sess:
        if FLAGS.model_state == 'train':
            print('------------Enter train model!------------')
            summary_writer = tf.summary.FileWriter(log_dir)
            for e in range(FLAGS.epoch):
                train_x, train_y = shuffle(train_x, train_y)
                for xs, ys in get_batches(train_x, train_y):
                    feed_dict = {
                        rnn_model.x: xs,
                        rnn_model.y_: ys
                    }
                    _, loss, step, merged_summary = sess.run(
                        [rnn_model.train_op, rnn_model.loss, rnn_model.global_step, rnn_model.merged_summary],
                        feed_dict=feed_dict)
                    if step % 10 == 0:
                        tf.logging.info('epoch->{} step->{} loss: {}'.format(e, step, loss))

                        summary_writer.add_summary(merged_summary, step)
                        if loss < minLoss:
                            minLoss = loss
                            saver.save(sess=sess, save_path=log_dir, global_step=step)
        if FLAGS.model_state == 'predict':
            print('------------Enter train model!------------')
            results = []

            for xs, ys in get_batches(test_x, test_y):
                feed_dict = {
                    rnn_model.x: xs,
                    rnn_model.y_: ys
                }
                predicts = sess.run(rnn_model.predicts, feed_dict=feed_dict)  # 不添加train_op这一优化操作，即不进行反向传播

                results.extend(predicts.tolist())

            plot_predicted = plt.plot(results, label="results", color='red')
            plot_test = plt.plot(test_y, label="real_sin", color='blue')
            plt.legend = ([plot_predicted, plot_test], ['predicted', 'real_sin'])
            mse = average_mse(test_y, results)
            tf.logging.info('average of mse: {}'.format(mse))
            plt.show()
