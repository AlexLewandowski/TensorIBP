
def khatrirao(matrices, reverse=False):
    # Compute the Khatri-Rao product of all matrices in list "matrices".
    # If reverse is true, does the product in reverse order.
    matorder = range(len(matrices)) if not reverse else list(reversed(range(len(matrices))))
    
    # Error checking on matrices; compute number of rows in result.
    # N = number of columns (must be same for each input)
    N = matrices[0].shape[1]
    # Compute number of rows in resulting matrix
    # After the loop, M = number of rows in result.
    M = 1
    for i in matorder:
        if matrices[i].ndim != 2:
            raise ValueError("Each argument must be a matrix.")
        if N != (matrices[i].shape)[1]:
            raise ValueError("All matrices must have the same number of columns.")
        M *= (matrices[i].shape)[0]
        
    # Computation
    # Preallocate result.
    P = NP.zeros((M, N))
    
    # n loops over all column indices
    for n in range(N):
        # ab = nth col of first matrix to consider
        ab = matrices[matorder[0]][:,n]
        # loop through matrices
        for i in matorder[1:]:
            # Compute outer product of nth columns
            ab = NP.outer(matrices[i][:,n], ab[:])
        # Fill nth column of P with flattened result
        P[:,n] = ab.flatten()
    return P


def im2double(im):
    info = NP.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(NP.float) / info.max



# In[7]:


def __get_unfolding_mode_order(A, n):
    return [i for i in xrange(n+1, A.ndim)] + [i for i in xrange(n)]
def __get_unfolding_stride(A, mode_order):
    stride = [0 for i in xrange(A.ndim)]
    stride[mode_order[A.ndim-2]] = 1
    for i in xrange(A.ndim-3, -1, -1):
        stride[mode_order[i]] = (
            A.shape[mode_order[i+1]] * stride[mode_order[i+1]])
    return stride
def __get_tensor_indices(r, c, A, n, mode_order, stride):
    i = [0 for j in xrange(A.ndim)]
    i[n] = r
    i[mode_order[0]] = c / stride[mode_order[0]]
    for k in xrange(1, A.ndim-1):
        i[mode_order[k]] = (
            (c % stride[mode_order[k-1]]) / stride[mode_order[k]])
    return i
def get_unfolding_matrix_size(A, n):
    row_count = A.shape[n]
    col_count = 1
    for i in xrange(A.ndim):
        if i != n: col_count *= A.shape[i]
    return (row_count, col_count)
def unfold(A, n):
    """
    Unfold tensor A along Mode n
    """
    (row_count, col_count) = get_unfolding_matrix_size(A, n)
    result = NP.zeros((row_count, col_count))
     
    mode_order = __get_unfolding_mode_order(A, n)
    stride = __get_unfolding_stride(A, mode_order)
         
    for r in xrange(row_count):
        for c in xrange(col_count):
            i = __get_tensor_indices(r, c, A, n, mode_order, stride)
            result[r, c] = A.__getitem__(tuple(i))
 
    return result