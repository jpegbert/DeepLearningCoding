import collections
import zipfile
import tensorflow as tf
import random
import numpy as np


# https://mp.weixin.qq.com/s/o7yWhLHVgg4E9ygn_YU4vA


max_vocabulary_size = 50000  # 语料库最大词语数
min_occurrence = 10  # 最小词频

# 读取数据
data_path = '../data/text8.zip'
with zipfile.ZipFile(data_path) as f:
    text_words = f.read(f.namelist()[0]).lower().split()  # 总共17005207词，我们只选取最常见的

# 创建计数器，从多到少计算词频
# 第一个词为‘UNK’，文本中不在语料库中的词汇用这个代替
count = [('UNK', -1)]
# 基于词频返回max_vocabulary_size个常用词
count.extend(collections.Counter(text_words).most_common(max_vocabulary_size - 1))

# 剔除掉出现次数少于'min_occurrence'的词
for i in range(len(count) - 1, -1, -1):
    if count[i][1] < min_occurrence:
        count.pop(i)
    else:
        break

# 每个词都分配一个ID
word2id = dict()
for i, (word, _)in enumerate(count):
    word2id[word] = i

# 所有词转换成ID
data = list()
unk_count = 0
for word in text_words:
    index = word2id.get(word, 0)
    if index == 0:
        unk_count += 1
    data.append(index)
count[0] = ('UNK', unk_count)
# id2word是word2id反映射
id2word = dict(zip(word2id.values(), word2id.keys()))


embedding_size = 200 # 词向量由200维向量构成
vocabulary_size = len(id2word) # 47135
embedding = tf.Variable(tf.random.normal([vocabulary_size, embedding_size])) # 维度：47135, 200


def get_embedding(x):
    """
    通过tf.nn.embedding_lookup函数将索引转换成词向量
    """
    x_embed = tf.nn.embedding_lookup(embedding, x)
    return x_embed


skip_window = 3 # 左右窗口大小
num_skips = 2 # 一次制作多少个输入输出对
batch_size = 128 #一个batch下有128组数据


def next_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # 数据窗口为7
    span = 2 * skip_window + 1
    # 创建队列，长度为7
    buffer = collections.deque(maxlen=span) # 创建一个长度为7的队列
    if data_index + span > len(data): # 如果文本被滑完，从头再来
        data_index = 0
    # 比如一个队列为deque([5234, 3081, 12, 6, 195, 2, 3134], maxlen=7)，数字代表词ID
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window] #上下文为[0, 1, 2, 4, 5, 6]
        words_to_use = random.sample(context_words, num_skips) # 在上下文里随机选2个候选词
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window] # 输入都为当前窗口的中间词，即3
            labels[i * num_skips + j, 0] = buffer[context_word] # 标签为当前候选词
        # 窗口右移，如果文本读完，从头再来
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


num_sampled = 64 # 负采样个数
# nce权重和偏差
nce_weights = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


def nce_loss(x_embed, y):
    """
    定义nce损失，x_emded为转化为词向量的中间词，y为上下文词
    """
    with tf.device('/cpu:0'):
        y = tf.cast(y, tf.int64)
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=y,
                           inputs=x_embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))
        return loss


with tf.GradientTape() as g:
    x, y = next_batch(batch_size, num_skips, skip_window)
    emb = get_embedding(x)
    loss = nce_loss(emb, y)

# 计算梯度
gradients = g.gradient(loss, [embedding, nce_weights, nce_biases])
# 更新
learning_rate = 0.01
optimizer = tf.optimizers.SGD(learning_rate)
optimizer.apply_gradients(zip(gradients, [embedding, nce_weights, nce_biases]))

