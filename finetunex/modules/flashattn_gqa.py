"""
supports GQA
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
    NUM_HEADS_Q: tl.constexpr,
    NUM_HEADS_K: tl.constexpr,
    SEQ_LEN_Q: tl.constexpr, 
    SEQ_LEN_K: tl.constexpr, 
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,

):
    block_index_q = tl.program_id(0) # indicates which block to process (remember that this is program id just like block id)
    #this indicate which batch and head to- process. each program is associated with a single head of a single batch. 
    idx_batch_head = tl.program_id(1)

    idx_batch = idx_batch_head // NUM_HEADS_Q # which batch this program is associated with
    idx_head_q = idx_batch_head % NUM_HEADS_Q # position of head in the batch

    GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_K
    idx_head_kv = idx_head_q // GROUP_SIZE
    # generate offset like - go to that particular batch and head in the tensor 
    # allows to get seqlen and head dim block in Q, K, V
    q_offset = (
        idx_batch.to(tl.int64) * stride_Q_batch
        + idx_head_q.to(tl.int64) * stride_Q_head
    )
    kv_offset = (
        idx_batch.to(tl.int64) * stride_K_batch
        + idx_head_kv.to(tl.int64) * stride_K_head
    )
    
    Q_block_ptr = tl.make_block_ptr(
        base = Q + q_offset, ## PARENT TENSOR , you are inside this tensor already (suppose head1 of batch1)# Q[index_batch, index_head, block_index_q, BLOCK_SIZE_:, :] block_index_q, BLOCK_SIZE_ says skipping some queries
        shape = (SEQ_LEN_Q, HEAD_DIM), # can say the whole block (a total blocks that has many blocks) you are accessing the elements from
        strides = (stride_Q_seq, stride_Q_dim),
        offsets = (block_index_q * BLOCK_SIZE_Q, 0), ## offsets define start row and start col , start col here is 0 means dont skip any col#we skip some queries right, we select the BLOCK (remember that blockwise computation!)
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1,0), ## for row-major 2-d tensor, (dim1, dim0) dim1 --> head dim (have smaller stride that is it has 1 stride). dim0 has longer stride, seq len is dim0 . (1,0) -> (smaller stride dim, longer stride dim)
    )
    
    V_block_ptr = tl.make_block_ptr(
        base = V + kv_offset, 
        shape = (SEQ_LEN_K, HEAD_DIM),
        strides = (stride_V_seq, stride_V_dim),
        offsets = (0, 0),
        block_shape = (BLOCK_SIZE_KV, HEAD_DIM),
        order = (1,0), 
    )
    
    K_block_ptr = tl.make_block_ptr(
        base = K + kv_offset, # K[index_batch, index_head,:, :] 
        shape = (HEAD_DIM,SEQ_LEN_K), #because K transpose 
        strides = (stride_K_dim, stride_K_seq),
        offsets = (0, 0),
        block_shape = (HEAD_DIM, BLOCK_SIZE_KV),
        order = (1,0), 
    )

    #output has same shape as query  # O[index_batch, index_head, block_index_q, BLOCK_SIZE_:, :]
    O_block_ptr = tl.make_block_ptr(
        base = O + q_offset, 
        shape = (SEQ_LEN_Q,HEAD_DIM), 
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
            SEQ_LEN_Q,
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
            SEQ_LEN_Q,
        )
        
        # need to compute logsumexp for backward pass. 
    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]
    m_ptrs = M + idx_batch_head * SEQ_LEN_Q + offset_q
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
    
    
import triton
import triton.language as tl
@triton.jit
def _attn_bwd_dk_dv(
    Q, K, V, Softmax_LSE, dO, 
    dK, dV, D, 
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,  
    stride_dob, stride_doh, stride_dom, stride_dok, 
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,  
    stride_deltab, stride_deltah, stride_deltam,  
    nheads_q: tl.constexpr, nheads_k: tl.constexpr,
    seqlen_q: tl.constexpr, seqlen_k: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale,
    BLOCK_Q: tl.constexpr, 
    BLOCK_KV: tl.constexpr,
    STAGE: tl.constexpr
):
    
    kv_blk_idx = tl.program_id(0)  # Which KV block (0 to seqlen_k//BLOCK_KV - 1)
    bid = tl.program_id(1)          # Which batch (0 to batch_size - 1)
    hkid = tl.program_id(2)         # Which KV head (0 to nheads_k - 1)

    
    group_size = nheads_q // nheads_k
    q_head_start = hkid * group_size
    q_head_end = q_head_start + group_size

    dk = tl.zeros([BLOCK_KV, head_dim], dtype=tl.float32)
    dv = tl.zeros([BLOCK_KV, head_dim], dtype=tl.float32)


    start_kv = kv_blk_idx * BLOCK_KV
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)
    offs_d = tl.arange(0, head_dim)
    
    # Mask for valid KV positions
    mask_kv = offs_kv < seqlen_k
    mask_kv_2d = mask_kv[:, None]
    
    # K pointer: [batch, kv_head, kv_seq, dim]
    k_ptrs = K + (bid * stride_kb + hkid * stride_kh) + \
             (offs_kv[:, None] * stride_kn + offs_d[None, :] * stride_kk)
    
    # V pointer: [batch, kv_head, kv_seq, dim]
    v_ptrs = V + (bid * stride_vb + hkid * stride_vh) + \
             (offs_kv[:, None] * stride_vn + offs_d[None, :] * stride_vk)
    
    # Load K and V tiles into SRAM
    k_tile = tl.load(k_ptrs, mask=mask_kv_2d, other=0.0)
    v_tile = tl.load(v_ptrs, mask=mask_kv_2d, other=0.0)

   
    for hqid in range(q_head_start, q_head_end):
        
        if STAGE == 3:  # Causal attention
            start_q_blk = start_kv // BLOCK_Q
        else:  
            start_q_blk = 0
        
       
        q_base = Q + (bid * stride_qb + hqid * stride_qh)
        do_base = dO + (bid * stride_dob + hqid * stride_doh)
        lse_base = Softmax_LSE + (bid * nheads_q + hqid) * seqlen_q
        d_base = D + (bid * stride_deltab + hqid * stride_deltah)

      
        num_q_blocks = tl.cdiv(seqlen_q, BLOCK_Q)
        
        for q_blk_idx in range(start_q_blk, num_q_blocks):
            start_q = q_blk_idx * BLOCK_Q
            offs_q = start_q + tl.arange(0, BLOCK_Q)
            
            # Mask for valid Q positions
            mask_q = offs_q < seqlen_q
            mask_q_2d = mask_q[:, None]
        
            q_ptrs = q_base + (offs_q[:, None] * stride_qm + offs_d[None, :] * stride_qk)
            do_ptrs = do_base + (offs_q[:, None] * stride_dom + offs_d[None, :] * stride_dok)
            
            q_tile = tl.load(q_ptrs, mask=mask_q_2d, other=0.0)
            do_tile = tl.load(do_ptrs, mask=mask_q_2d, other=0.0)
           
            lse = tl.load(lse_base + offs_q, mask=mask_q, other=0.0)
            
            d_vals = tl.load(d_base + offs_q * stride_deltam, mask=mask_q, other=0.0)

            s = tl.dot(q_tile, tl.trans(k_tile)) * softmax_scale
            
            p = tl.math.exp(s - lse[:, None])
            
            if STAGE == 3:  # Causal attention
                causal_mask = offs_q[:, None] >= offs_kv[None, :]
                p = tl.where(causal_mask, p, 0.0)

            dv += tl.dot(tl.trans(p.to(tl.float16)), do_tile.to(tl.float16))

            dp = tl.dot(do_tile.to(tl.float16), tl.trans(v_tile.to(tl.float16)))
           
            ds = p * (dp - d_vals[:, None])
            
            dk += tl.dot(tl.trans(ds.to(tl.float16)), q_tile.to(tl.float16)) * softmax_scale

    dk_ptrs = dK + (bid * stride_dkb + hkid * stride_dkh) + (offs_kv[:, None] * stride_dkn + offs_d[None, :] * stride_dkk)
    
    dv_ptrs = dV + (bid * stride_dvb + hkid * stride_dvh) + (offs_kv[:, None] * stride_dvn + offs_d[None, :] * stride_dvk)

    tl.store(dk_ptrs, dk, mask=mask_kv_2d)
    tl.store(dv_ptrs, dv, mask=mask_kv_2d)

@triton.jit
def _attn_bwd_dq(
    Q, K, V, softmax_scale, dO, dQ, M, D, 
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dob, stride_doh, stride_dom, stride_dok,
    NUM_HEADS_Q: tl.constexpr, 
    NUM_HEADS_K: tl.constexpr, 
    SEQ_LEN_Q: tl.constexpr, 
    SEQ_LEN_K: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    # Program IDs
    block_idx_q = tl.program_id(0)
    batch_idx = tl.program_id(1)
    hkid = tl.program_id(2)
    
    # Calculate Q head group
    GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_K
    q_head_start = hkid * GROUP_SIZE
    q_head_end = q_head_start + GROUP_SIZE
    
    # Adjust K/V to correct batch and KV head (shared across Q heads)
    adj_k = batch_idx * stride_kb + hkid * stride_kh
    adj_v = batch_idx * stride_vb + hkid * stride_vh
    K += adj_k
    V += adj_v
    
    # Offsets for this Q block
    offs_dim = tl.arange(0, HEAD_DIM)
    start_q = block_idx_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)
    
    # Masks and offsets for Q/dO loading
    mask_q = offs_q < SEQ_LEN_Q
    mask_q_2d = mask_q[:, None]
    
    # Delta for coordinate alignment
    delta_qk = SEQ_LEN_Q - SEQ_LEN_K
    
    # Loop over Q heads that share this KV head
    for hqid in range(q_head_start, q_head_end):
        # Adjust to correct Q head
        adj_q = batch_idx * stride_qb + hqid * stride_qh
        adj_do = batch_idx * stride_dob + hqid * stride_doh
        adj_delta = batch_idx * stride_deltab + hqid * stride_deltah
        adj_dq = batch_idx * stride_dqb + hqid * stride_dqh
        
        # Load Q and dO for this Q head
        q_ptrs = Q + adj_q + offs_q[:, None] * stride_qm + offs_dim[None, :] * stride_qk
        do_ptrs = dO + adj_do + offs_q[:, None] * stride_dom + offs_dim[None, :] * stride_dok
        
        Q_block = tl.load(q_ptrs, mask=mask_q_2d, other=0.0)
        dO_block = tl.load(do_ptrs, mask=mask_q_2d, other=0.0)
        
        # Load M and D
        m_ptrs = M + adj_delta + offs_q * stride_deltam
        d_ptrs = D + adj_delta + offs_q * stride_deltam
        
        M_block = tl.load(m_ptrs, mask=mask_q, other=0.0)
        M_block = M_block[:, None]  # [BLOCK_Q, 1]
        Di = tl.load(d_ptrs, mask=mask_q, other=0.0)  # [BLOCK_Q]
        
        # Initialize dQ accumulator
        dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
        
        # KV pointer setup
        offs_kv = tl.arange(0, BLOCK_KV)
        kT_ptrs = K + offs_kv[None, :] * stride_kn + offs_dim[:, None] * stride_kk
        vT_ptrs = V + offs_kv[None, :] * stride_vn + offs_dim[:, None] * stride_vk
        
        # Determine KV loop range
        if STAGE == 3:  # Causal
            hi = (block_idx_q + 1) * BLOCK_Q - delta_qk
            hi = min(max(hi, 0), SEQ_LEN_K)
            num_steps = (hi + BLOCK_KV - 1) // BLOCK_KV
        else:  # Non-causal
            hi = SEQ_LEN_K
            num_steps = SEQ_LEN_K // BLOCK_KV
        
        curr_kv = 0
        
        # Loop over KV blocks
        for blk_idx in range(num_steps):
            # Current KV indices
            offs_kv_curr = curr_kv + offs_kv
            
            # Mask for KV bounds
            mask_kv = offs_kv_curr < SEQ_LEN_K
            mask_kT = mask_kv[None, :]  # [1, BLOCK_KV]
            
            # Load K and V
            K_T_block = tl.load(kT_ptrs, mask=mask_kT, other=0.0)
            V_T_block = tl.load(vT_ptrs, mask=mask_kT, other=0.0)
            
            # Compute attention scores
            QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
            P_block = tl.math.exp(QK_block - M_block)
            
            # Apply causal mask if needed
            if STAGE == 3:
                causal_mask = (offs_q[:, None] - delta_qk) >= offs_kv_curr[None, :]
                P_block = tl.where(causal_mask, P_block, 0.0)
            
            # Compute gradients
            dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
            dS_block = P_block * (dP_block - Di[:, None])
            dS_block = dS_block.to(tl.float16)
            
            # Accumulate dQ
            dQ_block += tl.dot(dS_block, tl.trans(K_T_block))
            
            # Move to next KV block
            curr_kv += BLOCK_KV
            kT_ptrs += BLOCK_KV * stride_kn
            vT_ptrs += BLOCK_KV * stride_vn
        
        # Final scaling and store
        dQ_block *= softmax_scale
        
        dq_ptrs = dQ + adj_dq + offs_q[:, None] * stride_dqm + offs_dim[None, :] * stride_dqk
        tl.store(dq_ptrs, dQ_block, mask=mask_q_2d)
    
class TritonAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):

        batch_size, num_heads_q, seq_len_q, head_dim_q = Q.shape
        batch_size, num_heads_kv, seq_len_kv, head_dim_k = K.shape
        head_dim_v = V.shape[-1]
        # assert head_dim_q == head_dim_k and head_dim_k == head_dim_v
        #output where kernel saves output
        O = torch.empty_like(Q)
        flag = 3 if causal else 1

        grid = lambda args: (
            # (seqlen/blocksize) = total number of blocks of Q
            triton.cdiv(seq_len_q, args["BLOCK_SIZE_Q"]), #which group of queries are we going to work with
            batch_size * num_heads_q, #which head of which batch element are we going to work with
            1
        )
        #number of parallel programs: (Batch_size * num_heads * num_blocks_q)
        M = torch.empty(
            (batch_size, num_heads_q, seq_len_q), device = Q.device, dtype = torch.float32
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
            NUM_HEADS_Q = num_heads_q,
            NUM_HEADS_K = num_heads_kv,
            SEQ_LEN_Q = Q.shape[2], 
            SEQ_LEN_K = seq_len_kv,
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
        
        # Initialize gradients
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
    
        BATCH_SIZE, NUM_HEADS_Q, SEQ_LEN_Q, HEAD_DIM = Q.shape
        BATCH_SIZE, NUM_HEADS_K, SEQ_LEN_K, HEAD_DIM = K.shape
        
        NUM_WARPS, NUM_STAGES = 4, 1
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 64
    
        preprocess_grid = (SEQ_LEN_Q // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS_Q)
        D = torch.empty_like(M)  # Shape: (batch_size, num_heads_q, seq_len_q)
        
        _attn_bwd_preprocess[preprocess_grid](
            O=O, 
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN_Q,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=HEAD_DIM
        )

        stage = 3 if ctx.causal else 1
        
        grid_dk_dv = lambda meta: (
            triton.cdiv(SEQ_LEN_K, meta["BLOCK_KV"]), 
            BATCH_SIZE, 
            NUM_HEADS_K
        )
        grid_dq = lambda meta : (
            triton.cdiv(SEQ_LEN_Q, meta["BLOCK_Q"]),
            BATCH_SIZE,
            NUM_HEADS_K
        )
          
        _attn_bwd_dk_dv[grid_dk_dv](
            Q=Q,
            K=K,
            V=V,
            Softmax_LSE=M,
            dO=dO,
            dK = dK,
            dV = dV,
            D=D, 
          
            stride_qb=Q.stride(0),
            stride_qh=Q.stride(1),
            stride_qm=Q.stride(2),
            stride_qk=Q.stride(3),
            
            stride_kb=K.stride(0),
            stride_kh=K.stride(1),
            stride_kn=K.stride(2),
            stride_kk=K.stride(3),
           
            stride_vb=V.stride(0),
            stride_vh=V.stride(1),
            stride_vn=V.stride(2),
            stride_vk=V.stride(3),

            stride_dob=dO.stride(0),
            stride_doh=dO.stride(1),
            stride_dom=dO.stride(2),
            stride_dok=dO.stride(3),
            stride_dkb=dK.stride(0),
            stride_dkh=dK.stride(1),
            stride_dkn=dK.stride(2),
            stride_dkk=dK.stride(3),
            stride_dvb=dV.stride(0),
            stride_dvh=dV.stride(1),
            stride_dvn=dV.stride(2),
            stride_dvk=dV.stride(3),
            stride_deltab=D.stride(0),
            stride_deltah=D.stride(1),
            stride_deltam=D.stride(2),
            nheads_q=NUM_HEADS_Q,
            nheads_k=NUM_HEADS_K,
            seqlen_q=SEQ_LEN_Q,
            seqlen_k=SEQ_LEN_K,
            head_dim=HEAD_DIM,
            softmax_scale=ctx.softmax_scale,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        _attn_bwd_dq[grid_dq](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ = dQ,
            M=M,
            D=D, 
            # Q strides
            stride_qb=Q.stride(0),
            stride_qh=Q.stride(1),
            stride_qm=Q.stride(2),
            stride_qk=Q.stride(3),
            # K strides
            stride_kb=K.stride(0),
            stride_kh=K.stride(1),
            stride_kn=K.stride(2),
            stride_kk=K.stride(3),
            # V strides
            stride_vb=V.stride(0),
            stride_vh=V.stride(1),
            stride_vn=V.stride(2),
            stride_vk=V.stride(3),

            #dQ strides
            stride_dqb = dQ.stride(0), 
            stride_dqh = dQ.stride(1), 
            stride_dqm = dQ.stride(2), 
            stride_dqk = dQ.stride(3),
           
            stride_deltab=D.stride(0),
            stride_deltah=D.stride(1),
            stride_deltam=D.stride(2),

            stride_dob=dO.stride(0),
            stride_doh=dO.stride(1),
            stride_dom=dO.stride(2),
            stride_dok=dO.stride(3),
            NUM_HEADS_Q=NUM_HEADS_Q,
            NUM_HEADS_K=NUM_HEADS_K,
            SEQ_LEN_Q=SEQ_LEN_Q,
            SEQ_LEN_K=SEQ_LEN_K,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        return dQ, dK, dV, None, None