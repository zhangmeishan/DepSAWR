from configparser import ConfigParser
import sys, os
sys.path.append('..')

class Configurable(object):
    def __init__(self, config_file, extra_args):
        config = ConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        self._config = config
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        config.write(open(self.config_file, 'w'))
        print('Loaded config file sucessfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    @property
    def data_dir(self):
        return self._config.get('Data', 'data_dir')

    @property
    def train_files(self):
        return self._config.get('Data', 'train_files')

    @property
    def dev_files(self):
        return self._config.get('Data', 'dev_files')

    @property
    def test_files(self):
        return self._config.get('Data', 'test_files')

    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def save_src_vocab_path(self):
        return self._config.get('Save', 'save_src_vocab_path')

    @property
    def save_tgt_vocab_path(self):
        return self._config.get('Save', 'save_tgt_vocab_path')

    @property
    def load_dir(self):
        return self._config.get('Save', 'load_dir')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def load_src_vocab_path(self):
        return self._config.get('Save', 'load_src_vocab_path')

    @property
    def load_tgt_vocab_path(self):
        return self._config.get('Save', 'load_tgt_vocab_path')

    @property
    def model_name(self):
        return self._config.get('Network', 'model_name')

    @property
    def src_vocab_size(self):
        return self._config.getint('Network', 'src_vocab_size')

    @property
    def tgt_vocab_size(self):
        return self._config.getint('Network', 'tgt_vocab_size')

    @property
    def num_layers(self):
        return self._config.getint('Network', 'num_layers')

    @property
    def num_heads(self):
        return self._config.getint('Network', 'num_heads')

    @property
    def embed_size(self):
        return self._config.getint('Network', 'embed_size')

    @property
    def lstm_hidden_size(self):
        return self._config.getint('Network', 'lstm_hidden_size')

    @property
    def hidden_size(self):
        return self._config.getint('Network', 'hidden_size')

    @property
    def attention_size(self):
        return self._config.getint('Network', 'attention_size')

    @property
    def dropout_emb(self):
        return self._config.getfloat('Network', 'dropout_emb')

    @property
    def dropout_lstm_input(self):
        return self._config.getfloat('Network', 'dropout_lstm_input')

    @property
    def dropout_lstm_hidden(self):
        return self._config.getfloat('Network', 'dropout_lstm_hidden')

    @property
    def dropout_hidden(self):
        return self._config.getfloat('Network', 'dropout_hidden')

    @property
    def param_init(self):
        return self._config.getfloat('Network', 'param_init')

    @property
    def proj_share_weight(self):
        return self._config.getboolean('Network', 'proj_share_weight')

    @property
    def bridge_type(self):
        return self._config.get('Network', 'bridge_type')

    @property
    def label_smoothing(self):
        return self._config.getfloat('Network', 'label_smoothing')

    @property
    def learning_algorithm(self):
        return self._config.get('Optimizer', 'learning_algorithm')

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def decay_scale(self):
        return self._config.getfloat('Optimizer', 'decay_scale')

    @property
    def start_decay_at(self):
        return self._config.getint('Optimizer', 'start_decay_at')

    @property
    def decay_method(self):
        return self._config.get('Optimizer', 'decay_method')

    @property
    def lrate_decay_patience(self):
        return self._config.getint('Optimizer', 'lrate_decay_patience')

    @property
    def max_patience(self):
        return self._config.getint('Optimizer', 'max_patience')

    @property
    def min_lrate(self):
        return self._config.getfloat('Optimizer', 'min_lrate')

    @property
    def beta_1(self):
        return self._config.getfloat('Optimizer', 'beta_1')

    @property
    def beta_2(self):
        return self._config.getfloat('Optimizer', 'beta_2')

    @property
    def epsilon(self):
        return self._config.getfloat('Optimizer', 'epsilon')

    @property
    def clip(self):
        return self._config.getfloat('Optimizer', 'clip')


    @property
    def train_iters(self):
        return self._config.getint('Run', 'train_iters')

    @property
    def train_batch_size(self):
        return self._config.getint('Run', 'train_batch_size')

    @property
    def test_batch_size(self):
        return self._config.getint('Run', 'test_batch_size')

    @property
    def validate_every(self):
        return self._config.getint('Run', 'validate_every')

    @property
    def bleu_script(self):
        return self._config.get('Run', 'bleu_script')

    @property
    def save_after(self):
        return self._config.getint('Run', 'save_after')

    @property
    def eval_start(self):
        return self._config.getint('Run', 'eval_start')

    @property
    def update_every(self):
        return self._config.getint('Run', 'update_every')

    @property
    def decode_max_time_step(self):
        return self._config.getint('Run', 'decode_max_time_step')

    @property
    def beam_size(self):
        return self._config.getint('Run', 'beam_size')

    @property
    def max_src_length(self):
        return self._config.getint('Run', 'max_src_length')

    @property
    def max_tgt_length(self):
        return self._config.getint('Run', 'max_tgt_length')
