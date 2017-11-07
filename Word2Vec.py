# region 加载库,基础配置
import collections
import math
import os
import random
import zipfile
import numpy as np
#import urllib 
#python3中已经不能再用urllib了，应改为urllib.request
import urllib.request
import tensorflow as tf
url = 'http://mattmahoney.net/dc/'
vocabulary_size = 50000 # 最常用的单词总数设置在50000
embedding_size = 128  # 嵌入向量的维度
batch_size = 128      # 每次输入的inputs是128个单词,batch_size必须是num_skips的整数倍
num_skips = 2         # 用一个单词生成2个label,num_skips必须＜或=2×skip_window
skip_window = 1       # 只考虑左右各1个单词
# 验证样例，抽取频率最高的一部分单词，看看在向量空间上的分布情况。
valid_size = 16     # 取16个单词
valid_window = 100  # 在最高频的100个单词中选取
num_sampled = 64    # 负样本单词的数量，用作训练时的噪声
num_steps = 100001   # 训练步数
data_index = 0
# endregion

# region 1 数据预处理

# region 1.1 下载数据

# region 1.1.1 确认文件
# 确认文件的函数
# 通过传入文件名和文件大小,确认文件,否则下载后确认.确认后返回文件名.
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

# 确认文件.
# 传入文件名和大小，如果已经下载好了，就返回test8.zip给filename
filename = maybe_download('text8.zip', 31344016)
# endregion

# region 1.1.2 读文件
# 读文件内容的函数
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

# 读文件内容
# 传入filename=text8.zip，返回给单词的列表words，例如words=['i','love','apple','i','love',...,'back']
words = read_data(filename)
# 打印words列表的长度，即17005207个单词
print('words size', len(words))
# endregion

# endregion

# region 1.2 生成数据集

# region 1.2.1 数据集生成函数
# 传入单词列表words(形如words=['i','love','apple','i','love',...,'back']),返回以下4个数据集:

# 列表data，用于存放文本的单词顺序,形如data=[1,2,7,1,2,9,...]

# 列表count用于存放每个单词的频率,形如count=[['UNK',-1],('i',8),('love',5),...,('back',1)]

# 字典dictionary是英数词典，形如dictionary={'love':2,'i':1,'UNK':0,...}
# 字典dictionary可以根据单词查到对应数字，搭配words，可以生成data

# 字典reverse_dictionary是数英词典，形如reverse_dictionary={0:'UNK',1:'i',2:'love',...}
# 字典reverse_dictionary可以根据数字查到对应单词，搭配data，可以生成words
def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  #　经过上面这两句，count就成为一个拥有一堆列表和元组的列表，最常用的50000个单词（如果有）均被加入到列表count中，生僻单词则丢弃。
  #　第一个元素是一个2个元素的列表（存放罕见单词），后面所有元素都是一个2个元素的元组（即不可更改，第二个元素是单词的出现次数）
  #　例如count=[['UNK',-1],('i',8),('love',5),...,('back',1)]
  #　由于频率相同的两个单词，在每次形成的列表count中位置不尽相同，因此后面dictionary中相同频率单词（key）的value也会产生变化。
  #　建立字典，通过下面的循环，给每一个单词一个编号
  dictionary = dict()
  for word, _ in count:#第一轮，word=UNK，_=-1；第二轮，word=i，_=8；第三轮，word=love，_=5.....事实上只用到了word。
    dictionary[word] = len(dictionary)
  #　到这里，字典完工，其中的元素没有顺序。key（即单词）在count中的频率高，在count中的index就越小，则字典中的key的value就小。
  #　例如dictionary={'love':2,'i':1,'UNK':0,...}
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  #　上面的循环，根据字典dictionary里对每个词的编号（生僻单词编号为0，频率最高单词编号为1，依次为2），把列表words翻译成了列表data
  #　例如传入的words=['i','love','apple','i','love',...,'back'],则data=[1,2,7,1,2,9,...]
  count[0][1] = unk_count#即把-1变成了统计出来的具体数字，如count=[['UNK',3],('i',8),...]，就表示有3个生僻单词未使用。
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  #　把原来字典的每个元素的key和value互换了，英数词典变数英词典了。因为变成了数英词典，所以‘看起来’有了顺序
  #　例如reverse_dictionary={0:'UNK',1:'i',2:'love',...}
  return data, count, dictionary, reverse_dictionary
# endregion

# region 1.2.2 生成数据集,并检验效果
# 生成四个数据集
# data, count, dictionary, reverse_dictionary四个全局的数据集，可以被任何函数直接在内部拿来用，而不需要作为实参传入。
data, count, dictionary, reverse_dictionary = build_dataset(words)
# 删除words，节省内存
del words
# 列出头5个单词,第1个是生僻单词,后面4个是最常见单词Top4，
# 例如：[['UNK', 0], ('i', 3), ('love', 2), ('but', 1), ('stone', 1)]
print('Most common words (+UNK)', count[:5])
# 列出头10个data元素，然后由反转词典迭代翻译
# 输出形如
# Sample data [1, 2, 5, 1, 2, 7, 9, 3, 1, 6] ['i', 'love', 'apple', 'i', 'love', 'banana', 'too', 'but', 'i', "don't"]
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
# endregion

# endregion

# region 1.3 生成训练数据

# region 1.3.1 训练集数据生成函数
# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  #　单词序号data_index，因为需要反复使用，所以设置为全局变量。
  global data_index
  #　检验batch_size为num_skips整数倍，如果不是，报错。
  assert batch_size % num_skips == 0
  #　检验num_skips必须<=2×skip_window，如果不是，报错。
  assert num_skips <= 2 * skip_window
  #　初始化两个数组，batch，labels(labels是一个列向量数组)。
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 
  #　buffer是一个长度为span（即3）的双向队列
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  #　因为这里的span，导致data_index会以span倍数递增。以span=3为例，第一次出来的data_index是3，第二次出来是6依次类推。
  #　此时，buffer中存入了span个（即3个）的数，以第一次为例，buffer有三个数，包括data[0],data[1],data[2]。
  for i in range(batch_size // num_skips):#i=0,1,2,3
    #　i循环负责移动buffer，例如：
    #　i=0时，buffer是data[0],data[1],data[2]；i=1时，buffer是data[1],data[2],data[3]；以此类推。
    target = skip_window  
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):#j=0,1
      #　基于拿到的buffer，j循环负责生成两个batch元素和两个label元素。例如data=[1,2,9,4,2,8,...]
      #　当i=0时，buffer=[1,2,9]，j循环生成batch[0]=2,label[0]=1,batch[1]=2,label[1]=9.
      #　当i=1时，buffer=[2,9,4]，j循环生成batch[2]=9,label[2]=2,batch[3]=9,label[3]=4.以此类推
      #　即batch[x]和batch[x+1]中存着一个单词,则在label[x]和label[x+1]中的分别是这个单词前后的两个单词。
      #　用下面while+随机，使得label[x]和label[x+1]中存的单词，前后顺序随机。
      # 比如buffer[2,9,4]中，label[2]=4,label[3]=2。也可能是label[2]=2,label[3]=4
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    #　一个j循环结束，把data[3]存入队列buffer，例如第一次，buffer中的三个数变成了：data[1],data[2],data[3]。
    buffer.append(data[data_index])
    #　data_index递增一个数。
    data_index = (data_index + 1) % len(data)
    #　再次进入i循环
  return batch, labels
# endregion

# region 1.3.2 生成训练数据,并检验效果
# skip_window=1,表示每个单词只能与前一个单词和后一个单词联系，产生两个样本。
# 这里传入8,仅为了展示enerate_batch()的效果．真正的batch_size在上面配置．
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
# 输出的batch形如[2,2,9,9,4,4,2,2],labels形如[[1],[9],[2],[4],[9],[2],[4],[8]]

# 打印所有的batch元素与labels元素的对应表
# batch[i]的值就是一个数字，放到数英词典reverse_dictionary[]里，就是所对应的单词。labels同理。
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
      '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
# endregion

# endregion

# endregion

# region 2 构建计算图

# 在最热门的100个单词中随机拿出16个单词,组成形式上的拥有16个数字元素的列表.
# 只随机这一次，之后都是这个确定的valid_examples
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# 建立一个skip-gram model
graph = tf.Graph()
with graph.as_default():
  # region 定义输入操作
  # batch_size = 128      # 每次输入的inputs是128个单词
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

  # embeddings是所有单词随机生成的词向量，也就是权重，经过训练，正则化形成normalized_embeddings
  # embeddings相当于输入层到隐藏层的权重矩阵，也正是词嵌入训练后要拿到的最终成果．
  # vocabulary_size = 50000
  # embedding_size=128
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

  # 在embeddings里查找train_inputs（即128个单词）所对应的向量embed
  # train_inputs中有batch_size个数字(即代表128个单词的128列,形如[2,2,5,5,1,1,2,2,...]),
  # 在50000行embedings中,挑出这128行(即第2行,第2行,第5行...),依次排列,组成新的矩阵embed
  # embed的shape就是[batch_size,embedding_size]，[128,128]
  # 第一个128即行数来自train_inputs的列数(batch_size),第二个128来自embeddings的列数(embedding_size)
  # tf.nn.embedding_lookup()等于变相地进行了一次矩阵相乘计算．
  # 每一次train＿inputs都是不同的128个数字，因此每次embed从embeddings矩阵中抽出不同的128个行．
  embed = tf.nn.embedding_lookup(embeddings, train_inputs)
  # endregion

  # region 定义权重,loss,优化器
  # 为目标函数NCE Loss初始化权重和偏置
  # nce_weights相当于隐藏层到输出层的权重矩阵,训练结束后就没用了.
  nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
  nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels, # 传入的batch_labels,形如[[5],[1],[2],[5],[2],[1],...]
                     inputs=embed,        # 传入的train_inputs对embeddings处理后的[batch_size,embedding_size]
                     num_sampled=num_sampled,#即噪声数 让tf.nn.nec_loss()自动生成负样本
                     num_classes=vocabulary_size)# 类别数目，即在(vocabulary_size-1)里找num_sampled个负样本
                        )

  # SGD优化器，学习速率设为1.0
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
  # endregion

  # region 验证相似度
  # 传入随机采样的数字列表valid_examples,返回一个常量tensor
  # valid_examples形如[71 84 63 77 40 15  2  4 27 34 49 87 85  9 20 65]
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # 计算嵌入向量embeddings的模(L2范数norm)
  # 对[50000,128]的embeddings的每一行的元素的平方和再开根号,变成了[50000,1]的tensor的norm
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

  # embeddings除以范数norm就时标准化后的normalized_embeddings
  # normalized_embeddings就是[50000,128]的tensor,每一行有128个元素,它们的平方和为1
  # normalized_embeddings是词嵌入训练的终极结果，
  # 有了它，就可以计算任意词汇之间的相似度．或者进行可视化比对．
  normalized_embeddings = embeddings / norm

  # 使用tf.nn.embedding_lookup查询valid_dataset里单词的嵌入向量
  # 在normalized_embeddings里查找valid_dataset（即16个单词）所对应的向量valid_embeddings
  # valid_dataset中有16个数字(即16列,形如[71 84 63 77 ... 2 43]),
  # 在50000行normalized_embeddings中,挑出这16行(即第71行,第84行,第63行...),依次排列,组成新的矩阵valid_embeddings
  # valid_embeddings的shape就是[16,128]
  # 16个行数来自valid_dataset的列数(即16个随机单词),128来自normalized_embeddings的列数(embedding_size)
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)

  # 计算验证单词的嵌入向量与词汇表中的所有单词的相似性（余弦相似度）
  # 因为transpose_b=True,所以先对矩阵b转置,即把normalized_embeddings的[50000,128]转置为[128,50000]
  # [16,128]*[128,50000],得到[16,50000]的similarity
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
  # 定义所有变量的初始化tensor
  # endregion

  init = tf.global_variables_initializer()
# endregion

# region 3 执行计算图
with tf.Session(graph=graph) as session:
  # 初始化所有变量tensor
  init.run()
  print("Initialized")
  average_loss = 0
  for step in range(num_steps):
    # region 训练
    # batch_inputs, batch_labels才是真正的训练入口数据
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    # 通过我们之前定义的这个generate_batch()（前面已处理好的全局列表data在函数体内部就读入了），我们能得到一个batch的inputs和label。
    # batch_inputs形如[2,2,5,5,1,1,2,2,...], batch_labels形如[[5],[1],[2],[5],[2],[1],...]
    # 组成用来feed的字典
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    # 执行计算图中的训练
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)

    # 每一步（即每一个batch）都累加loss，用于后面的求平均loss
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # 过去2000步的（或曰2000个batch）执行的平均loss
      print("Average loss at step ", step, ": ", average_loss)
      # 打印了过去的平均loss，在进入下一步（下一个batch）前重新归零
      average_loss = 0
    # endregion

    # region 执行相似度计算
    # 每训练5000次，就计算一次验证单词与全部单词的相似度。
    # 并将与每个验证单词最相似的8个单词展示出来。
    if step % 5000 == 0:
      # sim是一个大小为[valid_size,vocabulary]的数组。sim[i,:]是valid_dataset[i]和其它元素的相似程度。
      sim = similarity.eval()
      # valid_size=16,就是说取16个单词,看看相似度.
      for i in range(valid_size):
        # 拿到valid_examples[i]的那个数字,在数英字典中找对应的那个单词,返回给valid_word
        # valid_examples形如[71 84 63 77 40 15  2  4 27 34 49 87 85  9 20 65]
        valid_word = reverse_dictionary[valid_examples[i]]
        # 组成打印序列
        log_str = "Nearest to %s:" % valid_word
        # 找和这个单词最接近的8个单词.
        top_k = 8
        # sim是[16,50000],sim[i,:]表示sim的第i行,即一个包含50000个数字元素的列表
        # sim[i,:].argsort()的结果是,把sim第i行的所有元素顺序排序.输出index数组,形如[...,103,408,132,71]共计50000个
        # (-sim[i,:]).argsort()就是逆序排列sim[i,:]中的所有index,形如[71,132,408,103,...],
        # (-sim[i, :]).argsort()[0]是valid_examples[i]本身，所以不输出
        # (-sim[i, :]).argsort()[1:top_k+1]就时取第1个数直到第(top_k+1)个数,结果为top_k个数的列表[132,408,...,]
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        # 打印输出最接近的8个单词
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          # 经过top_k次装填,lor_str就迭代成了一个"Nearest to ..."的单词序列
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
    # endregion

  # 画图中要用
  # final_embeddings就是进行了正则化后的从输入层到隐层的权重矩阵
  final_embeddings = normalized_embeddings.eval()
# endregion

# region 4 画图
# region 4.1 画图函数
# low_dim_embs是降维到2维的单词的空间向量，在图中展示每个单词的位置。
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  #　确认单词数不大于单词的向量列表
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"

  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    # 用plt.scatter显示散点图（即单词的位置）
    plt.scatter(x, y)
    # 用plt.annotate展示单词本身。
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
  # 保存图片到本地
  plt.savefig(filename)
# endregion

# region 4.2 画图
try:
  # TSNE用于降维。
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
  # 初始化tsne
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  # 只展示词频最高的100个单词
  plot_only = 100
  # 将[50000,128]的嵌入向量降到[100,128]
  # [:100,:]表示显示前100行,或者说从0行到第100行,列数则全部显示.详细见testmaohao.py
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
  # 通过数英词典,把数字翻译成单词列表labels
  labels = [reverse_dictionary[i] for i in range(plot_only)]
  # 用下面的函数进行展示
  plot_with_labels(low_dim_embs, labels)
# endregion

# region 4.3 异常处理
except ImportError:
  print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
# endregion
# endregion
