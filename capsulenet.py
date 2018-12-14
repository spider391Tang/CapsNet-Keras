"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import time
import wave
import librosa
import metrics
import numpy as np
from keras import layers, models, optimizers
import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images, create_folder, split_in_seqs, split_multi_channels
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from sklearn import preprocessing
from sklearn.metrics import f1_score

K.set_image_data_format('channels_last')


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def preprocess_data(_X, _Y, _X_test, _Y_test, _seq_len, _nb_ch):
    # split into sequences
    _X = split_in_seqs(_X, 1)
    #_Y = split_in_seqs(_Y, 1)

    _X_test = split_in_seqs(_X_test, 1)
    # _Y_test = split_in_seqs(_Y_test, 1)

    print "split multi channel: ", _X.shape
    _X = split_multi_channels(_X, _nb_ch)
    _X_test = split_multi_channels(_X_test, _nb_ch)
    _X = _X.reshape(-1, 1, 40, 1).astype('float32')
    _X_test = _X_test.reshape(-1, 1, 40, 1).astype('float32')

    # add extra label

    _Y_extra = np.zeros((_Y.shape[0],7))
    _Y_extra[:,:-1] = _Y
    for i in range( _Y.shape[0] ):
        if sum(_Y[i]) > 0:
            _Y_extra[i][6] = 0
        else:
            _Y_extra[i][6] = 1

    _Y_test_extra = np.zeros((_Y_test.shape[0],7))
    _Y_test_extra[:,:-1] = _Y_test
    for i in range( _Y_test.shape[0] ):
        if sum(_Y_test[i]) > 0:
            _Y_test_extra[i][6] = 0
        else:
            _Y_test_extra[i][6] = 1

    # np.apply_along_axis( addLabel, axis=1, arr=b )
    return _X, _Y_extra, _X_test, _Y_test_extra

def load_data(_feat_folder, _mono, _fold=None):
    feat_file_fold = os.path.join(_feat_folder, 'mbe_{}_fold{}.npz'.format('mon' if _mono else 'bin', _fold))
    dmp = np.load(feat_file_fold)
    _X_train, _Y_train, _X_test, _Y_test = dmp['arr_0'],  dmp['arr_1'],  dmp['arr_2'],  dmp['arr_3']
    return _X_train, _Y_train, _X_test, _Y_test

def load_audio(filename, mono=True, fs=44100):
    """Load audio file into numpy array
    Supports 24-bit wav-format
    
    Taken from TUT-SED system: https://github.com/TUT-ARG/DCASE2016-baseline-system-python
    
    Parameters
    ----------
    filename:  str
        Path to audio file

    mono : bool
        In case of multi-channel audio, channels are averaged into single channel.
        (Default value=True)

    fs : int > 0 [scalar]
        Target sample rate, if input audio does not fulfil this, audio is resampled.
        (Default value=44100)

    Returns
    -------
    audio_data : numpy.ndarray [shape=(signal_length, channel)]
        Audio

    sample_rate : integer
        Sample rate

    """

    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        _audio_file = wave.open(filename)

        # Audio info
        sample_rate = _audio_file.getframerate()
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels()
        number_of_frames = _audio_file.getnframes()

        # Read raw bytes
        data = _audio_file.readframes(number_of_frames)
        _audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.fromstring(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = np.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            audio_data = np.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate
    return None, None

def load_desc_file(_desc_file):
    _desc_dict = dict()
    for line in open(_desc_file):
        words = line.strip().split('\t')
        name = words[0].split('/')[-1]
        if name not in _desc_dict:
            _desc_dict[name] = list()
        _desc_dict[name].append([float(words[2]), float(words[3]), __class_labels[words[-1]]])
    return _desc_dict

def extract_mbe(_y, _sr, _nfft, _nb_mel):
    spec, n_fft = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=_nfft/2, power=1)
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel)
    return np.log(np.dot(mel_basis, spec))

def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=(3, 9), strides=1, padding='same', activation='relu', name='conv1')(x)
    #conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=(3, 9), strides=2, padding='same')
    #primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    best_epoch, pat_cnt, best_er, f1_for_best_er, best_conf_mat = 0, 0, 99999, None, None
    tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list = [0] * nb_epoch, [0] * nb_epoch, [0] * nb_epoch, [0] * nb_epoch
    posterior_thresh = 0.5

    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    hist = model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    print('Start testing ..<========')

    best_epoch, pat_cnt, best_er, f1_for_best_er, best_conf_mat = 0, 0, 99999, None, None
    tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list = [0] * nb_epoch, [0] * nb_epoch, [0] * nb_epoch, [0] * nb_epoch

    posterior_thresh = 0.7

    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=256)

    # print 'x_recon=', x_recon 
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    for i in range(13000,13500):
        print(y_pred[i], '|', y_test[i])

    # Calculate the predictions on test data, in order to calculate ER and F scores
    pred_thresh = y_pred > posterior_thresh
    # print('pred_thresh:', pred_thresh)
    score_list = metrics.compute_scores(pred_thresh, y_test, frames_in_1_sec=frames_1_sec)

    f1_overall_1sec_list = score_list['f1_overall_1sec']
    er_overall_1sec_list = score_list['er_overall_1sec']
    pat_cnt = pat_cnt + 1

    # y_test = y_test > posterior_thresh
    y_test = y_test.astype(int)
    pred_thresh = pred_thresh.astype(int)
    print('Shape y_test:', y_test.shape)
    print('Shape y_test:', y_test)
    print('Shape pred_thresh:', pred_thresh.shape)
    print('Shape pred_thresh:', pred_thresh)
    # Results
    print('sklearn Macro-F1-Score:', f1_score(y_test, pred_thresh, average='macro'))
    print('Custom Macro-F1-Score:', K.eval(f1(y_test, pred_thresh)))

    # np.set_printoptions(threshold=np.inf)
    # print y_pred

    # Calculate confusion matrix
    # test_pred_cnt = np.sum(pred_thresh, 2)
    # y_test_cnt = np.sum(Y_test, 2)
    # conf_mat = confusion_matrix(y_test_cnt.reshape(-1), test_pred_cnt.reshape(-1))
    # conf_mat = conf_mat / (utils.eps + np.sum(conf_mat, 1)[:, None].astype('float'))

    print('tr Er : {}, val Er : {}, F1_overall : {}, ER_overall : {} Best ER : {}, best_epoch: {}'.format(
        tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list, best_er, best_epoch))

    # img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    # image = img * 255
    # Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    # print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    # plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    # plt.show()


def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    print '==>', args.digit
    print x_test.shape
    print y_test.shape
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 6, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)
    print '===>', x_recon

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
 
    print "load_mnist"
    print x_train.shape
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    print x_train.shape
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    print y_train.shape
    y_train = to_categorical(y_train.astype('float32'))
    print y_train.shape
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # ###################################################################
    #              Main script starts here
    # ###################################################################

    is_mono = True
    __class_labels = {
        'brakes squeaking': 0,
        'car': 1,
        'children': 2,
        'large vehicle': 3,
        'people speaking': 4,
        'people walking': 5
    }

    # location of data.
    folds_list = [1] #[1, 2, 3, 4]
    # evaluation_setup_folder = '/Users/spider391tang/projects/TUT/TUT-sound-events-2017-development/evaluation_setup'
    # audio_folder = '/Users/spider391tang/projects/TUT/TUT-sound-events-2017-development/audio/street'
    # audio_folder = '/Users/spider391tang/projects/learn_feature'

    # Output
    # feat_folder = '/Users/spider391tang/projects/TUT/feat/'
    feat_folder = '/home/yhtseng/projects/data/feat/'
    create_folder(feat_folder)

    #######################################################################################
    # MAIN SCRIPT STARTS HERE
    #######################################################################################

    __fig_name = '{}_{}'.format('mon' if is_mono else 'bin', time.strftime("%Y_%m_%d_%H_%M_%S"))

    nb_ch = 1 if is_mono else 2
    batch_size = 128    # Decrease this if you want to run on smaller GPU's
    seq_len = 256       # Frame sequence length. Input to the CRNN.
    nb_epoch = 500      # Training epochs
    patience = int(0.25 * nb_epoch)  # Patience for early stopping

    # Number of frames in 1 second, required to calculate F and ER for 1 sec segments.
    # Make sure the nfft and sr are the same as in feature.py
    sr = 44100
    nfft = 2048
    frames_1_sec = int(sr/(nfft/2.0))

    print('\n\nUNIQUE ID: {}'.format(__fig_name))
    print('TRAINING PARAMETERS: nb_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}'.format(
        nb_ch, seq_len, batch_size, nb_epoch, frames_1_sec))

    # Folder for saving model and training curves
    __models_dir = 'models/'
    create_folder(__models_dir)

    # CRNN model definition
    cnn_nb_filt = 128            # CNN filter size
    cnn_pool_size = [5, 2, 2]   # Maxpooling across frequency. Length of cnn_pool_size =  number of CNN layers
    rnn_nb = [32, 32]           # Number of RNN nodes.  Length of rnn_nb =  number of RNN layers
    fc_nb = [32]                # Number of FC nodes.  Length of fc_nb =  number of FC layers
    dropout_rate = 0.5          # Dropout after each layer
    #print('MODEL PARAMETERS:\n cnn_nb_filt: {}, cnn_pool_size: {}, rnn_nb: {}, fc_nb: {}, dropout_rate: {}'.format( cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate))

    avg_er = list()
    avg_f1 = list()

    # User set parameters
    win_len = nfft
    hop_len = win_len / 2
    nb_mel_bands = 40

    X, Y, X_test, Y_test = load_data(feat_folder, is_mono, 1)
    print "number of channel: ", nb_ch 
    print "seq_len : ", seq_len

    x_train, y_train, x_test, y_test = preprocess_data(X, Y, X_test, Y_test, seq_len, nb_ch)

    # load data
    # (x_train, y_train), (x_test, y_test) = load_mnist()

    # define model
    print "input: ", x_train.shape[1:]
    # print "class: ", len(np.unique(np.argmax(y_train, 1)))

    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=7,
                                                  routings=args.routings)
    model.summary()

    print '==============summary done================='

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        # manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test), args=args)
