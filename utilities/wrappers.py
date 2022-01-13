import numpy as np
from utilities import io_utils


def slice_list_of_arrays(arrays, mask):
    if isinstance(arrays, tuple) | isinstance(arrays, list):
        return [slice_list_of_arrays(array, mask) for array in arrays]
    else:
        return arrays[mask]


def stack_list_of_arrays(arrays,padded=True):
    if isinstance(arrays[0], tuple) | isinstance(arrays[0], list):
        return [stack_list_of_arrays([array[k] for array in arrays],padded=padded) for k in range(len(arrays[0])) ]
    else:
        if padded:
            return np.concatenate(list(arrays), axis=0)
        else:
            return np.array(list(arrays), dtype=np.object)



def split_list_of_arrays(arrays, segment_lengths):
    nsplits = len(segment_lengths)
    split_indexes = [0] + list(np.cumsum(segment_lengths))
    if isinstance(arrays, tuple) | isinstance(arrays, list):
        return [split_list_of_arrays(array, segment_lengths) for array in arrays]
    else:
        return np.array([arrays[split_indexes[i]:split_indexes[i + 1]] for i in range(nsplits)])


def truncate_list_of_arrays(arrays, Ls):
    if isinstance(arrays, tuple) | isinstance(arrays, list):
        return [truncate_list_of_arrays(array, Ls) for array in arrays]
    else:
        if isinstance(Ls, list) | isinstance(Ls, np.ndarray):
            return np.array([array[:L] for array, L in zip(arrays, Ls)])
        else:
            return np.array([array[:Ls] for array in arrays])

def load_model(location, Lmax=None,load_weights=True):
    pickle_location = location + '.data'
    env = io_utils.load_pickle(pickle_location)
    backend = env['backend']
    if backend == 'sklearn':
        return env['model']
    elif backend == 'keras':
        wrapper_builder = env['wrapper_builder']
        try:
            wrapper_builder_kwargs = env['wrapper_builder_kwargs']
        except:
            wrapper_builder_kwargs = {}
        model_builder = env['model_builder']
        model_args = env['model_args']
        model_kwargs = env['model_kwargs']
        weights_location = location + '.h5'

        if (Lmax is not None):
            for keyword in model_kwargs.keys():
                if 'Lmax' in keyword:
                    model_kwargs[keyword] = None

            if 'Lmax' in model_kwargs.keys():
                model_kwargs['Lmax'] = Lmax
            if 'Lmax_aa' in model_kwargs.keys():
                model_kwargs['Lmax_aa'] = Lmax

            if ('Lmaxs' in wrapper_builder_kwargs.keys() ):
                Lmaxs_training = wrapper_builder_kwargs['Lmaxs']
                Lmaxs = []
                Lmax_training = Lmaxs_training[0]
                # Lmaxs = [int(np.ceil(Lmax/Lmaxs_training[0] * Lmax_)) for Lmax_ in Lmaxs_training]
                Lmaxs = [Lmax_ // Lmaxs_training[0] * Lmax + (Lmax_ % Lmaxs_training[0]) for Lmax_ in Lmaxs_training]


                wrapper_builder_kwargs['Lmaxs'] = Lmaxs


        model = wrapper_builder(
            model_builder, *model_args, **model_kwargs, backend=backend, **wrapper_builder_kwargs)
        if load_weights:
            model.model.load_weights(weights_location)
        return model
    else:
        raise ValueError('Backend not supported %s' % backend)
    return


class Predictor_wrapper():
    def __init__(self, model_builder, *args, backend='keras', **kwargs):
        self.wrapper_builder = type(self)
        self.model_builder = model_builder
        self.args = args
        self.kwargs = kwargs
        self.model = self.model_builder(*self.args, **self.kwargs)
        self.backend = backend

    def save(self, location):
        if self.backend == 'sklearn':
            pickle_location = location + '.data'
            io_utils.save_pickle(
                {'model': self, 'backend': self.backend}, pickle_location)
        elif self.backend == 'keras':
            pickle_location = location + '.data'
            weights_location = location + '.h5'

            env = {
                'backend': self.backend,
                'wrapper_builder': self.wrapper_builder,
                'model_builder': self.model_builder,
                'model_args': self.args,
                'model_kwargs': self.kwargs,
                'weights_location': weights_location
            }
            if hasattr(self,'wrapper_builder_kwargs'):
                env['wrapper_builder_kwargs'] = self.wrapper_builder_kwargs

            io_utils.save_pickle(env, pickle_location)

            self.model.save_weights(weights_location)
        else:
            raise ValueError('Backend not supported %s' % self.backend)

    def fit(self, *args,**kwargs):
        return self.model.fit(*args,**kwargs)

    def predict(self,*args,**kwargs):
        return self.model.predict(*args,**kwargs)

    def fit_generator(self, *args,**kwargs):
        return self.model.fit_generator(*args,**kwargs)

    def predict_generator(self,*args,**kwargs):
        return self.model.predict_generator(*args,**kwargs)


class point_Predictor_wrapper(Predictor_wrapper):
    def __init__(self, model_builder, *args, backend='sklearn', **kwargs):
        super(point_Predictor_wrapper, self).__init__(
            model_builder, *args, backend=backend, **kwargs)

    def fit(self, inputs, outputs, sample_weight=None, **kwargs):
        if sample_weight is not None:
            if isinstance(inputs, list) | isinstance(inputs, tuple):
                inputs_ = inputs[0]
            else:
                inputs_ = inputs
            lengthes = [len(input_) for input_ in inputs_]
            sample_weight = np.array([np.ones(
                length) * sample_weight_ for length, sample_weight_ in zip(lengthes, sample_weight)])
            sample_weight = stack_list_of_arrays(sample_weight)
        inputs = stack_list_of_arrays(inputs)
        outputs = stack_list_of_arrays(outputs)
        mask = (outputs==-1) | np.isnan(outputs) | (np.isinf(outputs)) | np.isinf(inputs).max(-1) | np.isnan(inputs).max(-1) # -1 = missing value.
        if mask.max():
            print('Discarding %s residues (missing label and/or nans)'%mask.sum() )
            inputs = inputs[~mask]
            outputs = outputs[~mask]
        sample_weight = sample_weight[~mask]
        return self.model.fit(inputs, outputs, sample_weight=sample_weight, **kwargs)

    def predict(self, inputs, Ls=None, return_all=False):

        if isinstance(inputs, tuple) | isinstance(inputs, list):
            inputs_ = inputs[0]
        else:
            inputs_ = inputs
        segment_lengths = [len(input) for input in inputs_]
        inputs = stack_list_of_arrays(inputs)
        try:
            # for sklearn object.
            outputs = self.model.predict_proba(inputs)[:, 1]
        except:
            # for keras objects.
            outputs = self.model.predict(inputs,verbose=True)[:, 0]

        outputs = split_list_of_arrays(outputs, segment_lengths)
        if not return_all:
            if isinstance(outputs, tuple) | isinstance(outputs, list):
                outputs = outputs[0]
        return outputs


class series_Predictor_wrapper(Predictor_wrapper):
    def __init__(self, model_builder, *args, backend='keras', **kwargs):
        super(series_Predictor_wrapper, self).__init__(
            model_builder, *args, backend=backend, **kwargs)

    def fit(self, inputs, outputs, **kwargs):
        return self.model.fit(inputs, outputs, **kwargs)

    def predict(self, inputs, Ls=None, return_all=False, truncated=True, batch_size=8):
        outputs = self.model.predict(inputs, batch_size=batch_size)

        if truncated:
            if Ls is None:
                if isinstance(inputs, list):
                    input_ = inputs[-1]
                else:
                    input_ = inputs
                Ls = input_.shape[1] - (input_ == 0.).min(-1).sum(-1)
            outputs = truncate_list_of_arrays(outputs, Ls)
        if return_all:
            return outputs
        else:
            if isinstance(outputs, list):
                output = outputs[0]
            else:
                output = outputs
            return np.array([output_[:, 1] for output_ in output])


class grouped_Predictor_wrapper(Predictor_wrapper):
    def __init__(self, model_builder, *args, backend='keras', multi_inputs=True,multi_outputs=False, verbose=True,
                 input_type = ['frames','points','attributes'],Lmaxs = [800,800,800],Lmax_outputs=None,**kwargs):
        super(grouped_Predictor_wrapper, self).__init__(
            model_builder, *args, backend=backend, **kwargs)
        self.multi_inputs = multi_inputs
        self.multi_outputs = multi_outputs
        self.input_type = input_type
        self.Lmax = Lmaxs
        if Lmax_outputs is None:
            self.Lmax_output = Lmaxs[0] if (isinstance(Lmaxs,list) | isinstance(Lmaxs,tuple)) else Lmaxs
        else:
            self.Lmax_output = Lmax_outputs

        self.big_distance = 3e3
        self.big_sequence_distance = 1000
        self.verbose=verbose
        self.wrapper_builder_kwargs = {'multi_inputs':self.multi_inputs,
                                       'multi_outputs':self.multi_outputs,
                                       'input_type':self.input_type,
                                       'Lmaxs':self.Lmax,
                                       'Lmax_outputs':self.Lmax_output,
                                       'verbose':self.verbose
                                       }

    def group_examples(self,Ls):
        Ls = np.array(Ls)
        if isinstance(self.Lmax,list):
            Lmax = self.Lmax[0]
        else:
            Lmax = self.Lmax

        order = np.argsort(Ls)[::-1]
        batches = []
        placed = np.zeros(len(Ls),dtype=np.bool)

        for k in order:
            if not placed[k]:
                if Ls[k]>= Lmax:
                    batches.append( [(k,0,Lmax)] )
                    placed[k] = True
                else:
                    current_batch = [(k,0, Ls[k] )]
                    placed[k] = True
                    batch_filled = False
                    current_batch_size = Ls[k]

                    while not batch_filled:
                        remaining_size = Lmax - current_batch_size
                        next_example = np.argmax(Ls  - 1e6 * ( placed + (Ls>remaining_size) )    )
                        if (Ls[next_example] <= remaining_size) and not placed[next_example]:
                            current_batch.append( (next_example, current_batch_size,current_batch_size+Ls[next_example] ) )
                            current_batch_size += Ls[next_example]
                            placed[next_example] = True
                        else:
                            batch_filled = True
                    batches.append(current_batch)
        permutation = np.argsort(np.random.rand(len(batches)))
        shuffled_batches = [batches[k] for k in permutation]
        return shuffled_batches



    def group_and_padd(self, inputs,groups,which='inputs',weights=None):
        ngroups = len(groups)
        if which == 'inputs':
            multi_valued = self.multi_inputs
            input_types = self.input_type
        else:
            multi_valued = self.multi_outputs
            input_types = 'outputs'

        if weights is not None:
            if multi_valued:
                Ls = np.array([len(input_) for input_ in inputs[0] ])
            else:
                Ls = np.array([len(input_) for input_ in inputs ])
            weights = weights/( (weights*Ls).mean()/ Ls.mean() )

            # weights = weights/weights.mean()


        if multi_valued:
            ninputs = len(inputs)
            grouped_inputs = []
            for n in range(ninputs):
                if isinstance(self.Lmax,list):
                    Lmax = self.Lmax[n]
                else:
                    Lmax = self.Lmax
                input_type = input_types[n]
                input_ = inputs[n]
                grouped_input = np.zeros([ngroups,Lmax]+list(input_[0].shape[1:]),dtype=input_[0].dtype)
                if input_type in ['indices','triplets']:
                    grouped_input += -1
                for k,group in enumerate(groups):
                    count = 0
                    start = 0
                    for example,_,_ in group:
                        end = min( start + len(input_[example]), Lmax)
                        if end-start>0:
                            grouped_input[k,start:end] = input_[example][:end-start]
                            if input_type == 'frames':
                                grouped_input[k,start:end,0,:] += count * self.big_distance
                            elif input_type =='points':
                                grouped_input[k,start:end] += count * self.big_distance
                            elif input_type == 'indices':
                                if count>0:
                                    grouped_input[k,start:end] += grouped_input[k,start-1] + self.big_sequence_distance
                            elif input_type == 'triplets':
                                if count>0:
                                    grouped_input[k,start:end] += grouped_input[k,:start].max()+1


                            elif (input_types == 'outputs') & (weights is not None):
                                grouped_input[k,start:end] *= weights[example]
                            start += min( len(input_[example]),Lmax)
                        else:
                            print(n,group,example,'Batch already filled; not enough space for this protein')
                        count +=1
                grouped_inputs.append(grouped_input)
        else:
            if isinstance(self.Lmax,list):
                Lmax = self.Lmax[0]
            else:
                Lmax = self.Lmax
            input_type = input_types
            grouped_inputs = np.zeros([ngroups,Lmax] + list(inputs[0].shape[1:]), dtype=np.float32)
            for k,group in enumerate(groups):
                count = 0
                for example,start,end in group:
                    grouped_inputs[k,start:end] = inputs[example][:Lmax]
                    if input_type == 'frames':
                        grouped_inputs[k,start:end,0,:] += count * self.big_distance
                    elif input_type == 'points':
                         grouped_inputs[k,start:end] += count * self.big_distance
                    elif input_type == 'indices':
                        if count>0:
                            grouped_inputs[k,start:end] += grouped_inputs[k,start-1] + self.big_sequence_distance
                    elif (input_type == 'outputs') & (weights is not None):
                        grouped_inputs[k,start:end] *= weights[example]
                    count +=1
        return grouped_inputs


    def ungroup_and_unpadd(self, grouped_outputs,groups,which='outputs'):
        if which == 'outputs':
            multi_valued = self.multi_outputs
        else:
            multi_valued = self.multi_inputs
        if multi_valued:
            nexamples = sum([len(group) for group in groups[0]])
            noutputs = len(grouped_outputs)
            outputs = [ np.array([None for _ in range(nexamples)],dtype=np.object) for _ in range(noutputs) ]
        else:
            nexamples = sum([len(group) for group in groups])
            outputs = np.array([None for _ in range(nexamples)],dtype=np.object)
            noutputs = 1
        if multi_valued:
            for n in range(noutputs):
                for k, group in enumerate(groups[n]):
                    for example, start, end in group:
                        outputs[n][example] = grouped_outputs[n][k][start:end]
        else:
            for k,group in enumerate(groups):
                for example,start,end in group:
                    outputs[example] = grouped_outputs[k][start:end]
        return outputs


    def predict(self, inputs,batch_size=8,return_all=False,truncated=True,Ls=None):
        if self.multi_inputs:
            Ls = [len(input_) for input_ in inputs[0] ]
            ninputs = len(inputs)
        else:
            Ls = [len(input_) for input_ in inputs]
            ninputs = 1
        if self.multi_outputs:
            noutputs = len(self.Lmax_output) if (isinstance(self.Lmax_output,list) | isinstance(self.Lmax_output,tuple) ) else 10
            if self.multi_inputs:
                if isinstance(self.Lmax_output,list) | isinstance(self.Lmax_output,tuple):
                    output2inputs = [self.Lmax.index(Lmax_output) for Lmax_output in self.Lmax_output]
                else:
                    output2inputs = [0 for _ in range(noutputs)]
                Loutputs = [[len(input_) for input_ in inputs[output2input]] for output2input in output2inputs]
            else:
                Loutputs = [Ls for _ in range(noutputs)]
        else:
            Loutputs = Ls
            noutputs = 1

        if self.verbose:
            print('Generating groups...')
        groups = self.group_examples(Ls)
        if self.multi_outputs:
            group_outputs = []
            for n in range(noutputs):
                Loutputs_ = Loutputs[n]
                Lmax_output = self.Lmax_output if isinstance(self.Lmax_output,int) else self.Lmax_output[n]
                group_outputs_ = []
                for group in groups:
                    start = 0
                    group_ = []
                    for index,_,_ in group:
                        group_.append( (index,min(start,Lmax_output), min(start+Loutputs_[index],Lmax_output) ) )
                        start += Loutputs_[index]
                    group_outputs_.append(group_)
                group_outputs.append(group_outputs_)
        else:
            group_outputs = groups


        if self.verbose:
            print('Grouped %s examples in %s groups'%(len(Ls),len(groups)) )
        if self.verbose:
            print('Grouping and padding...')
        grouped_inputs = self.group_and_padd(inputs,groups)
        if self.verbose:
            print('Performing prediction...')
        grouped_outputs = self.model.predict(grouped_inputs,batch_size=batch_size,verbose=True)


        if self.verbose:
            print('Ungrouping and unpadding...')
        outputs = self.ungroup_and_unpadd(grouped_outputs,group_outputs)

        if self.verbose:
            print('prediction done!')
        if (not return_all) & self.multi_outputs:
            return np.array([output_[:,1] for output_ in outputs[0]])
        elif (not return_all) & ~self.multi_outputs:
            return np.array([output_[:,1] for output_ in outputs])
        elif return_all & self.multi_outputs:
            return [np.array(outputs_) for outputs_ in outputs]
        else:
            return np.array([output_ for output_ in outputs])


    def fit(self, inputs, outputs, **kwargs):
        if self.multi_inputs:
            Ls = [len(input_) for input_ in inputs[0] ]
            ninputs = len(inputs)
        else:
            Ls = [len(input_) for input_ in inputs]
            ninputs = 1

        if 'sample_weight' in kwargs.keys():
            weights = kwargs.pop('sample_weight')
        else:
            weights = None

        if 'validation_data' in kwargs.keys():
            has_validation = True
            validation_data = kwargs.pop('validation_data')
            validation_inputs = validation_data[0]
            validation_outputs = validation_data[1]
            try:
                validation_weights = validation_data[2]
            except:
                validation_weights = None
            Lsvalidation = [len(input_) for input_ in validation_inputs[0] ]
        else:
            has_validation = False

        if self.verbose:
            print('Generating groups...')
        groups = self.group_examples(Ls)
        if has_validation:
            groups_validation = self.group_examples(Lsvalidation)

        if self.verbose:
            print('Grouped %s examples in %s groups'%(len(Ls),len(groups)) )
            if has_validation:
                print('Grouped %s validation examples in %s groups'%(len(Lsvalidation),len(groups_validation)) )
        if self.verbose:
            print('Grouping and padding...')
        grouped_inputs = self.group_and_padd(inputs,groups)
        grouped_outputs = self.group_and_padd(outputs,groups,which='outputs',weights=weights)
        if has_validation:
            grouped_validation_inputs = self.group_and_padd(validation_inputs,groups_validation)
            grouped_validation_outputs = self.group_and_padd(validation_outputs,groups_validation,which='outputs',weights=validation_weights)
            kwargs['validation_data'] = (grouped_validation_inputs,grouped_validation_outputs)


        if self.verbose:
            print('Fitting...')
        history = self.model.fit(grouped_inputs,grouped_outputs, **kwargs)
        if self.verbose:
            print('Fitting done!')
        return history


#%% Testing: