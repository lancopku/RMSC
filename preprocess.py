import argparse
import codecs
import json
import jieba
import os
import logging
import pickle
from sklearn import metrics
from sklearn.metrics import hamming_loss
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
# from bn_lstm import LSTMCell, BNLSTMCell
# from model_components import task_specific_attention, bidirectional_rnn, relation, soft_max_with_t
import tensorflow.contrib.layers as layers
try:
  from tensorflow.contrib.rnn import RNNCell
except ImportError:
  RNNCell = tf.nn.rnn_cell.RNNCell
try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
try:
    from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, DropoutWrapper
except ImportError:
    MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell
    GRUCell = tf.nn.rnn_cell.GRUCell


class Configs(object):
    def __init__(self):
        # ------parsing input arguments"--------
        parser = argparse.ArgumentParser()

        parser.add_argument('--data_dir', type=str, default="data/small")
        parser.add_argument('--save_name', type=str, default="rmsc")
        parser.add_argument('--random_seed', type=int, default=1)
        parser.add_argument('--use_bn', type=bool, default=False)
        parser.add_argument('--gpu', type=str, default='1')
        parser.add_argument('--kp', type=float, default=1.0)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--use_birnn', type=bool, default=True)
        parser.add_argument('--cut_prob', type=float, default=0.2,
                            help='output a label if its probability is above cut_prob')

        parser.add_argument('--epoch_len', type=int, default=100)
        parser.add_argument('--start_eval_epoch', type=int, default=20)
        parser.add_argument('--optim', type=str, default='adam')
        parser.add_argument('--learning_rate', type=float, default=1e-3, )
        parser.add_argument('--decay_rate', type=float, default=1,
                            help='')
        parser.add_argument('--decay_steps', type=int, default=1000,
                            help='')

        # label graph
        parser.add_argument('--use_rel',  default=False, type=bool, help='whether use label graph ')
        parser.add_argument('--pre_rel', default=False, type=bool,
                            help='whether use label graph before sigmoid')
        parser.add_argument('--lg_init_noise', type=float, default=0,
                            help='add noise to the identity matrix during the label graph initialization')
        parser.add_argument('--ratio2', type=float, default=0,
                            help=' rel loss rate aggregate the label graph by adding  -||G-I||^2 to the loss function')

        # soft training
        parser.add_argument('--soft_target', default=False, type=bool,
                            help='whether use a continuous target as the second supervisor')
        parser.add_argument('--stop_gradient', default=True,
                            help="whether stop gradient (y'= stop_gradient(y*G))")
        parser.add_argument('--ratio1', type=float, default=1.0, help='loss += ratio1 * soft_target_loss')
        parser.add_argument('--temperature', type=int, default=0,
                            help='whether use temperature to soften the continuous target, default 0 is not to soft')

        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()
        #  ---- to member variables -----
        for key, value in self.args.__dict__.items():
            if key not in ['test', 'shuffle']:
                exec('self.%s = self.args.%s' % (key, key))


cfg = Configs()


# label graph
use_rel=cfg.use_rel
pre_rel = cfg.pre_rel
ratio2 = cfg.ratio2
lg_init_noise = cfg.lg_init_noise

# second signal
soft_target = cfg.soft_target
ratio1 = cfg.ratio1  # soft target loss
stop_gradient = cfg.stop_gradient
temperature = cfg.temperature

# base hyper params
random_seed = cfg.random_seed
learning_rate = cfg.learning_rate
use_bn = cfg.use_bn
cut_prob = cfg.cut_prob
data_dir = cfg.data_dir
epoch_len = cfg.epoch_len
start_eval_epoch = cfg.start_eval_epoch
kp = cfg.kp
sentence_out_size = 100
word_level_out = 100
units_num = 128
use_birnn = cfg.use_birnn
batch_size = cfg.batch_size
comments_num = 40
embed_dim = 100

# save dir
save_name = cfg.save_name

save_name = save_name
if cut_prob != 0.2:
    save_name += "_prob-"+str(cut_prob)
if learning_rate != 0.001:
    save_name += "_lr-" + str(learning_rate)
if cfg.optim != 'adam':
    save_name += '_optim-' + str(cfg.optim)
if cfg.random_seed != 1:
    save_name += '_seed-' + str(random_seed)
if cfg.decay_rate < 1:
    save_name += '_dr-' + str(cfg.decay_rate)
    save_name += '_ds-' + str(cfg.decay_steps)
if use_rel:
    save_name += "_ur"
    if pre_rel:
        save_name += "_pr"
    if lg_init_noise != 0:
        save_name += "_n-" + str(lg_init_noise)
    if ratio2 > 0:
        save_name += '_r2-' + str(ratio2)
    if soft_target:
        save_name += '_st'
        save_name += "_r1-" + str(ratio1)
        if temperature > 0:
            save_name += "_t-" + str(temperature)
        if not stop_gradient:
            save_name += "_nst"
save_dir = os.path.join("checkpoints", save_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
f_log = codecs.open(os.path.join(save_dir, 'log.txt'), 'w+', encoding='utf-8')

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
tf.set_random_seed(random_seed)
np.random.seed(random_seed)
songs = os.listdir(data_dir)
total_tags = []
all_comments_list = []
total_song_comments_and_tags = []
len_songs = len(songs)

for song in songs:
    with codecs.open(data_dir+'/'+song, 'r+', encoding='utf-8') as f:
        a = f.read()
        dict_f = json.loads(a)
        every_song_comment_for_lstm_train = []
        all_comments_dict = dict_f['all_short_comments'][0:40]
        for comment_dict in all_comments_dict:
            every_comment = comment_dict["comment"]
            every_comment_cut = "/".join(jieba.cut(every_comment, cut_all=False)).split('/')    # cur comment
            every_song_comment_for_lstm_train.append(every_comment_cut)     # [[pl1],[pl2],...]
        every_song_comments_and_tags = {"tags": dict_f['tags'], "comments": every_song_comment_for_lstm_train}
        all_comments_list.extend(every_song_comment_for_lstm_train)     # get all comments
        total_tags.extend(dict_f['tags'])  # get all classes
        total_song_comments_and_tags.append(every_song_comments_and_tags)  # all tags and comments
print(len(songs))
print(len(total_song_comments_and_tags))
sorted_total_tags = sorted(list(set(total_tags)))   # all tags
del total_tags
tags_num = len(sorted_total_tags)

# word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = Word2Vec(all_comments_list, workers=4, min_count=1, seed=random_seed)
model.save("data/word2vec.model")
del all_comments_list
print("model info :", model)

# map
dict_tag = {}
for i in range(tags_num):
    dict_tag[sorted_total_tags[i]] = i+1


def give_embedding(word):
    return model[word]


def tag_emb(tag):
    return dict_tag[tag]


# labels_origin = np.zeros((len_songs, 8))
labels_origin = []
sentence_long = 20
embed_input = np.zeros((len_songs, comments_num, sentence_long, embed_dim))
# get input and label
# word embedding
len_every_comment_cutted = np.zeros((len_songs, comments_num))
sum_comment_count = 0
for i in range(len_songs):
    every_song_comment_words_embedding = np.zeros((comments_num, sentence_long, embed_dim))
    all_comments_in_every_song = total_song_comments_and_tags[i]["comments"]
    for j in range(len(all_comments_in_every_song)):
        every_comment_in_one_song = np.array(list(map(give_embedding, all_comments_in_every_song[j])))
        every_comment_embedding = np.zeros((sentence_long, embed_dim))
        count_cur_comment = len(every_comment_in_one_song)
        sum_comment_count += count_cur_comment
        if count_cur_comment < sentence_long:
            len_every_comment_cutted[i, j] = count_cur_comment
            every_comment_embedding[0:count_cur_comment] = every_comment_in_one_song
        else:
            every_comment_embedding = every_comment_in_one_song[0:sentence_long]
            len_every_comment_cutted[i, j] = sentence_long
            # print(count_cur_comment)
        every_song_comment_words_embedding[j] = every_comment_embedding
    embed_input[i] = every_song_comment_words_embedding
    every_song_comment_tags_embedding = np.array(list(map(tag_emb, total_song_comments_and_tags[i]["tags"])))
    labels_origin.append(every_song_comment_tags_embedding)
del total_song_comments_and_tags
print("average comment length", sum_comment_count/(len_songs*comments_num))
labels = MultiLabelBinarizer().fit_transform(labels_origin)
del labels_origin
de_rate = 0
soft_labels = (1-de_rate) * labels + de_rate*(1-labels)
print(labels)
len_labels = len(labels)
# split the data
x_train, x_test_valid, y_train, y_test_valid, sequence_train, sequence_test_valid, songs_train, songs_test_valid = \
    train_test_split(embed_input, soft_labels, len_every_comment_cutted, songs, test_size=0.3, random_state=random_seed)
del soft_labels
del embed_input
del len_every_comment_cutted
x_test, x_valid, y_test, y_valid, sequence_test, sequence_valid, songs_test, songs_valid = train_test_split(x_test_valid, y_test_valid, sequence_test_valid, songs_test_valid, test_size=0.3, random_state=random_seed)
label_train, label_test_valid = train_test_split(labels, test_size=0.3, random_state=random_seed)
label_test, label_valid = train_test_split(label_test_valid, test_size=0.3, random_state=random_seed)


f_label_train = open('data/label_train.pickle', 'wb')
pickle.dump(label_train, f_label_train)
f_label_valid = open('data/label_valid.pickle', 'wb')
pickle.dump(label_valid, f_label_valid)
f_label_test = open('data/label_test.pickle', 'wb')
pickle.dump(label_test, f_label_test)

f_x_train = open('data/x_train.pickle', 'wb')
pickle.dump(x_train, f_x_train)
f_x_valid = open('data/x_valid.pickle', 'wb')
pickle.dump(x_valid, f_x_valid)
f_x_test = open('data/x_test.pickle', 'wb')
pickle.dump(x_test, f_x_test)

f_y_train = open('data/y_train.pickle', 'wb')
pickle.dump(y_train, f_y_train)
f_y_valid = open('data/y_valid.pickle', 'wb')
pickle.dump(y_valid, f_y_valid)
f_y_test = open('data/y_test.pickle', 'wb')
pickle.dump(y_test, f_y_test)

f_sorted_total_tags = open('data/sorted_total_tags.pickle', 'wb')
pickle.dump(sorted_total_tags, f_sorted_total_tags)

f_sequence_train = open('data/sequence_train.pickle', 'wb')
pickle.dump(sequence_train, f_sequence_train)
f_sequence_valid = open('data/sequence_valid.pickle', 'wb')
pickle.dump(sequence_valid, f_sequence_valid)
f_sequence_test = open('data/sequence_test.pickle', 'wb')
pickle.dump(sequence_test, f_sequence_test)

f_songs_test = open('data/songs_test.pickle', 'wb')
pickle.dump(songs_test, f_songs_test)
f_songs_valid = open('data/songs_valid.pickle', 'wb')
pickle.dump(songs_valid, f_songs_valid)
f_songs_train = open('data/songs_train.pickle', 'wb')
pickle.dump(songs_train, f_songs_train)

