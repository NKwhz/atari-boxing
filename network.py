import tensorflow as tf
import numpy as np

ACTION_SIZE = 18
STATE_SIZE = 128
BATCH_SIZE = 32
GAMMA = 0.9

LAYER_SIZES = [64, 32, ACTION_SIZE]

class AgentNetwork:
    def __init__(self, gamma=GAMMA, action_size=ACTION_SIZE, state_size=STATE_SIZE, batch_size=BATCH_SIZE, layers=LAYER_SIZES):
        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = batch_size
        self.layer_sizes = layers

        self.input = None
        self.output = None
        self.loss = None
        self.optimizer = None
        self.y = None
        self.a = None

        self.build_network()

        self.gamma = gamma

    def build_network(self):
        self.input = tf.placeholder(tf.float32, (None, self.state_size), name="s")
        x = self.input
        for i in range(len(self.layer_sizes)):
            if i < len(self.layer_sizes) - 1:
                x = tf.layers.dense(x, self.layer_sizes[i])
                x = tf.nn.relu(x)
            else:
                x = tf.layers.dense(x, self.layer_sizes[i], name="out")
            self.output = x

        self.y = tf.placeholder(tf.float32, (None, self.action_size), name='y')
        self.a = tf.placeholder(tf.int32, (None, self.action_size), name='a')

        y_ = tf.reduce_sum(tf.multiply(self.output, self.a), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.y - y_))
        self.optimizer = tf.train.AdamOptimizer()

    # data should be shuffled
    def train(self, data, sess):
        sess.run(tf.initialize_all_variables())

        # data loading
        s, a, r, s_, t = data
        available = s.shape[0]
        # available = int(s.shape[0] // self.batch_size * self.batch_size)
        # s_plh = tf.placeholder(tf.float32, s.shape)
        # a_plh = tf.placeholder(tf.int32, a.shape)
        # r_plh = tf.placeholder(tf.float32, r.shape)
        # _s_plh = tf.placeholder(tf.float32, s_.shape)
        # dataset = tf.data.Dataset.from_tensor_slices((s_plh, a_plh, r_plh, _s_plh))
        # iterator = dataset.make_initializable_iterator()
        # sess.run(iterator.initializer, feed_dict={
        #     s_plh: s,
        #     a_plh: a,
        #     r_plh: r,
        #     _s_plh: s_
        # })

        start = 0
        while start < available:
            if available - start >= self.batch_size:
                batch_size = self.batch_size
            else:
                batch_size = available - start
            s_batch = s[start : start + batch_size]
            a_batch = a[start : start + batch_size]
            r_batch = r[start : start + batch_size]
            _s_batch = s_[start : start + batch_size]
            t_batch = t[start : start + batch_size]
            _q_batch = sess.run(self.output, feed_dict={"s:0" : _s_batch})
            y_batch = []
            for i in range(batch_size):
                if t_batch[i]:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + self.gamma * np.max(_q_batch[i]))
            y_batch = np.stack(y_batch)

            sess.run(self.optimizer, feed_dict={
                "y:0" : y_batch,
                "s:0" : s_batch,
                "a:0" : a_batch
            })

            start += self.batch_size

    def predict(self, s, sess, return_q=True):     
        if return_q:
            """
            sess.run returns a array of network outputs, 
            if a one-line return is expected, 
            remember to pick it out from the array with [0]
            """
            return sess.run(self.output, feed_dict={"s:0" : s})
        else:
            return np.argmax(sess.run(self.output, feed_dict={"s:0" : s}), axis=1)