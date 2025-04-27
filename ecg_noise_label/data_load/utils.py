import numpy as np
from numpy.testing import assert_array_almost_equal



# basic function
def multiclass_noisify(y, P, random_state=123):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = int(y[idx])
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y



def noisify_multiclass_symmetric(y_train, noise, random_state=123, nb_classes=2):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
    # print(P)
    else:
        y_train_noisy = y_train
        actual_noise = 0.0
    return y_train_noisy, actual_noise



def noisify_adb_asymmetric(y_train, noise, random_state=123):
    """mistakes:
        L -> N
        V -> R
    """
    nb_classes = max(y_train) + 1
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # L -> N
        P[0, 0], P[0, 1] = 1. - n, n

        # V -> R
        P[3, 3], P[3, 2] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.4f' % actual_noise)
        y_train = y_train_noisy

    return y_train, actual_noise



def noisify(y_train, num_class, noise_type, noise_rate, seed):
    num_class = int(num_class)
    actual_noise = 0.0
    if noise_type == 'symmetric':
        y_train, actual_noise = noisify_multiclass_symmetric(y_train, noise_rate, random_state=seed, nb_classes=num_class)
    elif noise_type == 'asymmetric':
        y_train, actual_noise = noisify_adb_asymmetric(y_train, noise_rate, random_state=seed)

    return y_train, actual_noise