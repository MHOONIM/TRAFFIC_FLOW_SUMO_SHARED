import numpy as np
import tensorflow as tf

def tensorConversion(input):
    '''
    Tensor conversion method \\
    This method will convert any format of input to be a tensor format (float32)
    (In order to perform tensorflow operations, variables must be in tensor format)
    The output dimension will also be changed from 1dim (dim,) to 2dims (1, dims)

    Input Argument: Any (constant, list or array)
    Return: Input in tensor format with shape [N, dims]
    '''
    input = np.array(input)  # Convert the input to array first (to be able to find ndim.)
    if input.ndim < 2:
        input = np.reshape(input, [1, len(input)])
    elif input.ndim > 2:
        raise('Too many dimension for the data')
    return tf.convert_to_tensor(input, dtype=tf.float32)
