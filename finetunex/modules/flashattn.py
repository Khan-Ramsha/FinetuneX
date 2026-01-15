"""
This is the Flash attention implementation that supports SEQLEN multiple of 128 & equal number of Query, Key, Value heads. 

"""


import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner (
    O_block,
    l_i, 
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale, 
    BLOCK_SIZE_Q: tl.constexpr, 
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offset_q: tl.constexpr,
    offset_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    #range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        #used only for the block in which there is a transition between non-masked and masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q,(block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        #only used for non casual attn. 
        lo, hi = 0, SEQ_LEN
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    #loop over k, v and update the accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        #just let the compiler know that start_n is the multiple of block_n so the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
        
        #load the current k block and do dot product
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        #multiply by softmax scale 
        #QK_block *= softmax_scale

        if STAGE == 2:
            # we are ON the diagonals
            mask = offset_q[:, None] >= (start_kv + offset_kv[None, :])
            QK_block  = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            #compute the max value of qk or keep the old max val
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        #compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)

        #compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1)

        #this is the correction factor for previous l_i
        alpha = tl.math.exp(m_i - m_ij) #m_ij coming from the correct block

        #apply the correction factor
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)

        #this computes the following: O_new = P * V + O_old * alpha
        O_block = O_block * alpha[:, None] #fix the prev block
        O_block = tl.dot(P_block, V_block, O_block) # O_bloc += P_block dotproduct V_Block
        m_i = m_ij

        #Move to the next block of K and V
        K_block_ptr = tl.advance(K_block_ptr, (0,BLOCK_SIZE_KV))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
    return O_block, l_i, m_i


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages = num_stages,
            num_warps = num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key = ["SEQ_LEN", "HEAD_DIM"],
)          

@triton.jit
def _attn_fwd(
    Q, # not really a tensor, its a pointer to the first element of that tensor in the memory
    K, 
    V,
    softmax_scale,
    M, 
    O,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head, 
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE, 
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr, 
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,

):
    block_index_q = tl.program_id(0) # indicates which block to process (remember that this is program id just like block id)
    #this indicate which batch and head to- process. each program is associated with a single head of a single batch. 
    idx_batch_head = tl.program_id(1)

    idx_batch = idx_batch_head // NUM_HEADS # which batch this program is associated with
    idx_head = idx_batch_head % NUM_HEADS # position of head in the batch

    # generate offset like - go to that particular batch and head in the tensor 
    # allows to get seqlen and head dim block in Q, K, V
    qvk_offset = (
        idx_batch.to(tl.int64) * stride_Q_batch
        + idx_head.to(tl.int64) * stride_Q_head
    )
    
    Q_block_ptr = tl.make_block_ptr(
        base = Q + qvk_offset, ## PARENT TENSOR , you are inside this tensor already (suppose head1 of batch1)# Q[index_batch, index_head, block_index_q, BLOCK_SIZE_:, :] block_index_q, BLOCK_SIZE_ says skipping some queries
        shape = (SEQ_LEN, HEAD_DIM), # can say the whole block (a total blocks that has many blocks) you are accessing the elements from
        strides = (stride_Q_seq, stride_Q_dim),
        offsets = (block_index_q * BLOCK_SIZE_Q, 0), ## offsets define start row and start col , start col here is 0 means dont skip any col#we skip some queries right, we select the BLOCK (remember that blockwise computation!)
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1,0), ## for row-major 2-d tensor, (dim1, dim0) dim1 --> head dim (have smaller stride that is it has 1 stride). dim0 has longer stride, seq len is dim0 . (1,0) -> (smaller stride dim, longer stride dim)
    )
    
    V_block_ptr = tl.make_block_ptr(
        base = V + qvk_offset, 
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_V_seq, stride_V_dim),
        offsets = (0, 0),
        block_shape = (BLOCK_SIZE_KV, HEAD_DIM),
        order = (1,0), 
    )
    
    K_block_ptr = tl.make_block_ptr(
        base = K + qvk_offset, # K[index_batch, index_head,:, :] 
        shape = (HEAD_DIM,SEQ_LEN), #because K transpose 
        strides = (stride_K_dim, stride_K_seq),
        offsets = (0, 0),
        block_shape = (HEAD_DIM, BLOCK_SIZE_KV),
        order = (1,0), 
    )

    #output has same shape as query  # O[index_batch, index_head, block_index_q, BLOCK_SIZE_:, :]
    O_block_ptr = tl.make_block_ptr(
        base = O + qvk_offset, 
        shape = (SEQ_LEN,HEAD_DIM), 
        strides = (stride_O_seq, stride_O_dim),
        offsets = (block_index_q * BLOCK_SIZE_Q, 0),
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1,0), 
    )
    ### notes: parallelize along with each query blocks, then a forloop for values and keys. 
    #Offset for the token in the q to process. 
    offset_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    #the offsets of the tokens in the K and V sequence to process 
    offset_kv = tl.arange(0, BLOCK_SIZE_KV)

    #initialize the variables (read the paper) mi, li, ouptut (before a for loop)
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype = tl.float32) - float("inf") # so we initialize it with -inf 

    #li is the running sum, we have one for each queries as we sum the attention scores by rows.  
    l_i  = tl.zeros([BLOCK_SIZE_Q], dtype = tl.float32) + 1.0

    # li and mi are the statistics used. 

    #the accumulator for the output, which is group of rows of O
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype = tl.float32)
 # in causal attn we dont let query to attend keys that comes after it but in non-casual attention we let query to attend to all the keys 

    Q_block = tl.load(Q_block_ptr) #load blocks of Q from HBM to SRAM
    
    
    if STAGE == 3 or STAGE == 1:
        #this step run for the non-causal attn or for the left block of the diagonal in the casual attention. 
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block, 
            l_i, 
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale, 
            BLOCK_SIZE_Q, 
            BLOCK_SIZE_KV,
            4 - STAGE,
            offset_q,
            offset_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # run the function for right part of the diagonal
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block, 
            l_i, 
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale, 
            BLOCK_SIZE_Q, 
            BLOCK_SIZE_KV,
            2,
            offset_q,
            offset_kv,
            SEQ_LEN,
        )
        
        # need to compute logsumexp for backward pass. k
    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]
    m_ptrs = M + idx_batch_head * SEQ_LEN + offset_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))
    

@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D, 
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q) # 33, 34. 35, 36
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)

    #load a single block of block size q rows of 0
    O_block = tl.load(
        O
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
        
    ).to(tl.float32) #(BLOCK_SIZE_Q, head_dim)

    #load a single block of block size q rows of dO
    
    dO_block = tl.load(
        
        dO
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :] 
    ).to(tl.float32)

    D_block = tl.sum(dO_block * O_block, axis = 1) # not a matmul, but elementwise multiplication. (one scalar value for each row, the scalar value is sum of whole row.)
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)
    

@triton.jit
def _attn_bwd_dk_dv(
    Q, K, V, softmax_scale, dO, dQ, dK, dV, M, D,
    stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS, SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    M += offset_batch_head_seq
    D += offset_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    K_block = tl.load(K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    V_block = tl.load(V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim)

    # Determine loop range
    if STAGE == 3:  # Causal
        lo = index_block_kv * BLOCK_KV  # start_kv
        hi = SEQ_LEN
        num_steps = (hi - lo) // BLOCK_Q
        curr_q = lo
    else:
        lo = 0
        hi = SEQ_LEN
        num_steps = SEQ_LEN // BLOCK_Q
        curr_q = 0

    offs_q = tl.arange(0, BLOCK_Q)
    
    # Initialize pointers at position curr_q (not 0!)
    qT_ptrs = Q + (curr_q + offs_q[None, :]) * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + (curr_q + offs_q[:, None]) * stride_seq + offs_dim[None, :] * stride_dim

    for blk_idx in range(num_steps):
        qT_block = tl.load(qT_ptrs)
        offs_q_curr = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q_curr)

        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        if STAGE == 3:
            mask_block = offs_q_curr[None, :] >= offs_kv[:, None]
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        dO_block = tl.load(dO_ptrs)
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        Di = tl.load(D + offs_q_curr)
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))

        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)


@triton.jit
def _attn_bwd_dq(
    Q, K, V, softmax_scale, dO, dQ, dK, dV, M, D,
    stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS, SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    M += offset_batch_head_seq
    D += offset_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_q = tl.program_id(0)
    start_q = index_block_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)

    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]
    Di = tl.load(D + offs_q)

    offs_kv = tl.arange(0, BLOCK_KV)
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim


    curr_kv = 0
    # num_steps = SEQ_LEN // BLOCK_KV
    if STAGE == 3:  # Causal
    # Only iterate over K/V that current Q block can attend to
        lo = 0
        hi = (index_block_q + 1) * BLOCK_Q  # start_q + BLOCK_Q
        num_steps = (hi + BLOCK_KV - 1) // BLOCK_KV
    else:
        lo = 0
        hi = SEQ_LEN
        num_steps = SEQ_LEN // BLOCK_KV


    for blk_idx in range(num_steps):
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block)

        if STAGE == 3:
            offs_kv_curr = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv_curr[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))

        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq

    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)
    
class TritonAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        head_dim_q, head_dim_k = Q.shape[-1], K.shape[-1]
        head_dim_v = V.shape[-1]
        batch_size, num_heads, seq_len, head_dim = Q.shape

        assert head_dim_q == head_dim_k and head_dim_k == head_dim_v
        #output where kernel saves output
        O = torch.empty_like(Q)
        flag = 3 if causal else 1

        grid = lambda args: (
            # (seqlen/blocksize) = total number of blocks of Q
            triton.cdiv(seq_len, args["BLOCK_SIZE_Q"]), #which group of queries are we going to work with
            batch_size * num_heads, #which head of which batch element are we going to work with
            1
        )
        #number of parallel programs: (Batch_size * num_heads * num_blocks_q)
        M = torch.empty(
            (batch_size, num_heads, seq_len), device = Q.device, dtype = torch.float32
        )

        _attn_fwd[grid](
            Q = Q,
            K = K, 
            V = V,
            softmax_scale = softmax_scale,
            M = M, 
            O = O,
            #strides are used for accessing the elements because usually we get pointer to first element.
            stride_Q_batch = Q.stride(0),
            stride_Q_head = Q.stride(1),
            stride_Q_seq = Q.stride(2),
            stride_Q_dim = Q.stride(3),
            stride_K_batch = K.stride(0),
            stride_K_head = K.stride(1),
            stride_K_seq = K.stride(2),
            stride_K_dim = K.stride(3),
            stride_V_batch = V.stride(0),
            stride_V_head = V.stride(1),
            stride_V_seq = V.stride(2),
            stride_V_dim = V.stride(3),
            stride_O_batch = O.stride(0),
            stride_O_head = O.stride(1),
            stride_O_seq = O.stride(2),
            stride_O_dim = O.stride(3),
            BATCH_SIZE = Q.shape[0], 
            NUM_HEADS = Q.shape[1],
            SEQ_LEN = Q.shape[2], 
            HEAD_DIM = head_dim_k,
            STAGE = flag

        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = head_dim_k
        ctx.causal = causal
        return O

#you need to do TritonAttn.apply(pass key query values here)

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors
        if not dO.is_contiguous():
            dO = dO.contiguous()
        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        
        #store results
        #initialize dQ, dK, dV
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
    

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 1
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 64

        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M) #Shape: (batch_size, num_heads, seq_len)

        # Computing all the elements Di
        _attn_bwd_preprocess[preprocess_grid](
            O = O, 
            dO = dO,
            D = D,
            SEQ_LEN = SEQ_LEN,
            BLOCK_SIZE_Q = BLOCK_SIZE_MACRO,
            HEAD_DIM = ctx.HEAD_DIM
        )

        # grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)
        stage = 3 if ctx.causal else 1

        #fix kv and iterate through all the Q blocks
        grid_dk_dv = lambda meta: (
            triton.cdiv(SEQ_LEN, meta["BLOCK_KV"]), 1, BATCH_SIZE * NUM_HEADS
        )
    
        grid_dq = lambda meta: (
            triton.cdiv(SEQ_LEN, meta["BLOCK_Q"]), 1, BATCH_SIZE * NUM_HEADS
        )
        
        _attn_bwd_dk_dv[grid_dk_dv](
            Q = Q,
            K = K,
            V = V,
            softmax_scale = ctx.softmax_scale,
            dO = dO,
            dQ = dQ,
            dK = dK,
            dV = dV,
            M = M,
            D = D,
            stride_batch = Q.stride(0),
            stride_head = Q.stride(1),
            stride_seq = Q.stride(2),
            stride_dim = Q.stride(3),
            NUM_HEADS = NUM_HEADS,
            SEQ_LEN = SEQ_LEN,
            BLOCK_Q = BLOCK_SIZE_MICRO,
            BLOCK_KV = BLOCK_SIZE_MACRO,
            HEAD_DIM = ctx.HEAD_DIM,
            STAGE = stage,
            num_warps = NUM_WARPS,
            num_stages = NUM_STAGES,
            
        )

        # fix Q and iterate through all kv blocks
        _attn_bwd_dq[grid_dq](
            Q = Q,
            K = K,
            V = V,
            softmax_scale = ctx.softmax_scale,
            dO = dO,
            dQ = dQ,
            dK = dK,
            dV = dV,
            M = M,
            D = D,
            stride_batch = Q.stride(0),
            stride_head = Q.stride(1),
            stride_seq = Q.stride(2),
            stride_dim = Q.stride(3),
            NUM_HEADS = NUM_HEADS,
            SEQ_LEN = SEQ_LEN,
            BLOCK_Q = BLOCK_SIZE_MACRO,
            BLOCK_KV = BLOCK_SIZE_MICRO,
            HEAD_DIM = ctx.HEAD_DIM,
            STAGE = stage,
            num_warps = NUM_WARPS,
            num_stages = NUM_STAGES,
            
        )
        return dQ, dK, dV, None, None