import h5py
import numpy as np
import pickle


def get_from_dataset(f, key, verbose=True):
    if verbose:
        print('Loading %s' % key)
    isList = f[key + '/' + 'isList'].value
    isDictionary = f[key + '/' + 'isDictionary'].value
    isArray = f[key + '/' + 'isArray'].value
    if isDictionary:
        keys = f[key + '/' + 'listKeys'].value.astype('U')
        out = {}
        for key_ in keys:
            out[key_] = get_from_dataset(f, key + '/' + key_, verbose=verbose)
    elif isArray:
        itemType = f[key + '/' + 'itemType'].value
        if itemType == 'object':
            lenItem = f[key + '/' + 'lenItem'].value
            out = np.array([get_from_dataset(
                f, key + '/' + 'subitems/%s' % k, verbose=verbose) for k in range(lenItem)])
        else:
            out = f[key + '/' + 'data'].value.astype(itemType)
        if isList:
            out = list(out)
    else:
        out = f[key + '/' + 'data'].value
    return out


def add_to_dataset(f, key, item, verbose=True):
    isDictionary = isinstance(item, dict)
    isList = isinstance(item, list) | isinstance(item, tuple)
    if isList:
        item = np.array(item)
    isArray = type(item) == np.ndarray

    f[key + '/' + 'isList'] = isList
    f[key + '/' + 'isDictionary'] = isDictionary
    f[key + '/' + 'isArray'] = isArray

    if isDictionary:
        keys = list(item.keys())
        f[key + '/' + 'listKeys'] = np.array(keys, dtype='S')
        for key_ in keys:
            add_to_dataset(f, key + '/' + key_, item[key_], verbose=verbose)
    elif isArray:
        itemType = str(item.dtype)
        f[key + '/' + 'itemType'] = itemType
        if itemType == 'object':
            lenItem = len(item)
            f[key + '/' + 'lenItem'] = lenItem
            if verbose:
                print('%s is of type object, saving subitems' % key)
            for k, item_ in enumerate(item):
                add_to_dataset(f, key + '/' + 'subitems/%s' %
                               k, item_, verbose=verbose)
        else:
            if 'U' in itemType:
                item = item.astype('S')
            if verbose:
                print('Adding %s to dataset' % key)
            f.create_dataset(key + '/' + 'data', data=item)
    else:
        if verbose:
            print('Adding %s to dataset' % key)
        f.create_dataset(key + '/' + 'data', data=item)
    return


def save_h5py(env, filename, verbose=True, subset=None, exclude=None):
    if subset is not None:
        keys = subset
    else:
        keys = list(env.keys())
    if exclude is not None:
        for l, key in enumerate(keys):
            if key in exclude:
                del keys[l]

    with h5py.File(filename, 'w', libver='latest') as f:
        add_to_dataset(f, 'listKeys', keys, verbose=verbose)
        for key in keys:
            add_to_dataset(f, key, env[key], verbose=verbose)
    return


def load_h5py(filename, verbose=True, subset=None, exclude=None):
    env = {}
    with h5py.File(filename, 'r', libver='latest') as f:
        if subset is not None:
            keys = subset
        else:
            keys = get_from_dataset(f, 'listKeys', verbose=verbose)
        if exclude is not None:
            for l, key in enumerate(keys):
                if key in exclude:
                    del keys[l]
        for key in keys:
            env[key] = get_from_dataset(f, key, verbose=verbose)
    return env


def load_pickle(filename, subset=None, exclude=None):
    env = pickle.load(open(filename, 'rb'))
    if (subset is not None) | (exclude is not None):
        if subset is not None:
            keys = subset
        else:
            keys = list(env.keys())
        if exclude is not None:
            for l, key in enumerate(keys):
                if key in exclude:
                    del keys[l]

        env_ = dict([(key, env[key]) for key in keys])
    else:
        env_ = env
    return env_


def save_pickle(env, filename, subset=None, exclude=None, protocol=4):
    if (subset is not None) | (exclude is not None):
        if subset is not None:
            keys = subset
        else:
            keys = list(env.keys())
        if exclude is not None:
            for l, key in enumerate(keys):
                if key in exclude:
                    del keys[l]

        env_ = dict([(key, env[key]) for key in keys])
    else:
        env_ = env
    pickle.dump(env_, open(filename, 'wb'), protocol)
    return


def load_pickle_splitted(filename, nsubsets, subset_indexes=None, subset=None, exclude=None):
    assert nsubsets > 1
    if subset_indexes is None:
        subset_indexes = range(1, nsubsets + 1)
    else:
        if not isinstance(subset_indexes, list):
            subset_indexes = [subset_indexes]

    list_filenames = [filename[:-5] + '_%s_%s.data' %
                      (index, nsubsets) for index in subset_indexes]
    env = load_pickle(list_filenames[0], subset=subset, exclude=exclude)
    for filename in list_filenames[1:]:
        env2 = load_pickle(filename, subset=subset, exclude=exclude)
        for key in env.keys():
            if isinstance(env[key], list):
                env[key] += env2[key]
            elif isinstance(env[key], np.ndarray):
                env[key] = np.concatenate((env[key], env2[key]), axis=0)
            elif isinstance(env[key], dict):
                for key_ in env[key].keys():
                    if isinstance(env[key][key_], list):
                        env[key][key_] += env2[key][key_]
                    elif isinstance(env[key][key_], np.ndarray):
                        env[key][key_] = np.concatenate(
                            (env[key][key_], env2[key][key_]), axis=0)
    return env


def write_labels(list_origins, list_sequences, list_resids, list_labels, output_file):
    nexamples = len(list_origins)
    with open(output_file, 'w') as f:
        for n in range(nexamples):
            origin = list_origins[n]
            sequence = list_sequences[n]
            label = list_labels[n]
            resids = list_resids[n]
            L = len(sequence)
            f.write('>%s\n' % origin)
            for l in range(L):
                if label.dtype == np.float:
                    f.write('%s %s %s %.4f\n' % (resids[l, 0], resids[l, 1], sequence[l], label[l]))
                else:
                    f.write('%s %s %s %s\n' % (resids[l, 0], resids[l, 1], sequence[l], label[l]))
    return output_file

def read_labels(input_file, nmax=np.inf, label_type='int'):
    list_origins = []
    list_sequences = []
    list_labels = []
    list_resids = []

    with open(input_file, 'r') as f:
        count = 0
        for line in f:
            if (line[0] == '>'):
                if count == nmax:
                    break
                if count > 0:
                    list_origins.append(origin)
                    list_sequences.append(sequence)
                    list_labels.append(np.array(labels))
                    list_resids.append(np.array(resids))

                origin = line[1:-1]
                sequence = ''
                labels = []
                resids = []
                count += 1
            else:
                line_splitted = line[:-1].split(' ')
                resids.append(line_splitted[:2])
                sequence += line_splitted[2]
                if label_type == 'int':
                    labels.append(int(line_splitted[-1]))
                else:
                    labels.append(float(line_splitted[-1]))

    list_origins.append(origin)
    list_sequences.append(sequence)
    list_labels.append(np.array(labels))
    list_resids.append(np.array(resids))

    list_origins = np.array(list_origins)
    list_sequences = np.array(list_sequences)
    list_labels = np.array(list_labels)
    list_resids = np.array(list_resids)
    return list_origins, list_sequences, list_resids, list_labels
