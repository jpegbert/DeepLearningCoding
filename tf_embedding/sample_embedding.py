import tensorflow as tf
import numpy as np


def sample_embedding_example():
    """
    采用tensorflow实现embedding效果
    :return:
    """
    embedding = np.random.random([10, 6])
    inputs = np.array([[2, 3], [1, 4], [1, 2], [1, 3]], dtype=np.int32)
    embedding_layer = tf.nn.embedding_lookup(embedding, inputs)
    sess = tf.Session()
    print(sess.run([embedding_layer]))


def extend_embedding_example():
    """
    采用tensorflow实现embedding效果
    由tensorflow随机生成一个矩阵(单词,向量表),起到和上面一个函数一样的效果
    :return:
    """
    vocab_size = 5
    emb_size = 6
    embedding = tf.get_variable("embedding", [vocab_size, emb_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
    inputs = np.array([[2, 3], [1, 4], [1, 2], [1, 3]], dtype=np.int32)
    embedding_layer = tf.nn.embedding_lookup(embedding, inputs)
    sess = tf.Session()
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    print(sess.run([embedding_layer]))


if __name__ == '__main__':
    # sample_embedding_example()
    extend_embedding_example() # 与上面那种效果一样，只是采用了tf中的初始化方法

