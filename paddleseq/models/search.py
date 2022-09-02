import paddle

class BeamSearch(object):
    def __init__(self, ):
        super().__init__()
        self.constraint_states = None
        self.stop_on_max_len = False

    def step(
            self,
            step: int,
            lprobs,  # [bsz,beam,vocab_size]
            scores,  # [bsz,beam,step] /None when step=0
            prev_output_tokens=None,
            original_batch_idxs=None,
    ):
        bsz, beam_size, vocab_size = lprobs.shape
        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :]  # [bsz,1,vocab_size] 每个样本beam个里面选一个
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        top_prediction = paddle.topk(
            lprobs.reshape((bsz, -1)),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.reshape((bsz, -1)).shape[1] - 1,  # -1 so we never select pad
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        # Project back into relative indices and beams
        beams_buf = indices_buf // vocab_size  # 整数，得到第几个beam
        indices_buf = indices_buf.mod(paddle.to_tensor(vocab_size, dtype='int64'))  # 得到余数，是索引

        # At this point, beams_buf and indices_buf are single-dim and contain relative indices
        return scores_buf, indices_buf, beams_buf






class Sampling(object):
    sampling_topk: int
    sampling_topp: float

    def __init__(self, sampling_topk=-1, sampling_topp=-1.0):
        super().__init__()
        self.sampling_topk = sampling_topk
        self.sampling_topp = sampling_topp
        self.stop_on_max_len = False

    def _sample_topp(self, lprobs):
        """Sample among the smallest set of elements whose cumulative probability mass exceeds p.

        See `"The Curious Case of Neural Text Degeneration"
        (Holtzman et al., 2019) <https://arxiv.org/abs/1904.09751>`_.

        Args:
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step

        Return: A tuple of (trimed_probs, truncated_indices) where:
            trimed_probs: (bsz x input_beam_size x ?)
                the model's probabilities over the elements selected to sample from. The
                width of the third dimension is determined by top-P.
            truncated_indices: (bsz x input_beam_size x ?)
                the indices of the chosen elements.
        """
        probs = paddle.exp(lprobs)

        # sort the last dimension (vocab dimension) in descending order
        # sorted_probs, sorted_indices = probs.sort(descending=True)
        sorted_probs = paddle.sort(probs, descending=True)
        sorted_indices = paddle.argsort(probs, descending=True)

        # compute a mask to indicate the words to be included in the top-P set.
        cumsum_probs = sorted_probs.cumsum(axis=2)
        mask = cumsum_probs <= self.sampling_topp  # [bsz,beam=1,vocab_size]

        # note that mask was computed by 'lt'. One more word needs to be included
        # so that the cumulative probability mass can exceed p.
        bsz,beam,vocab_size=probs.shape
        cumsum_mask = paddle.cumsum(mask, axis=2, dtype="int64") # [bsz,beam=1,vocab_size] ,int64
        last_included = cumsum_mask[:, :, -1:] #  [bsz,beam=1,1]
        last_included = paddle.clip(last_included, 0, mask.shape[2] - 1)

        mask = paddle.cast(mask, dtype="int64").reshape([-1,vocab_size])
        mask= scatter(tensor=mask,dim=1,index=last_included.squeeze(-1),src=paddle.ones_like(mask))
        # 结果
        mask=paddle.cast(mask.reshape([bsz, beam, vocab_size]),dtype="bool")
        # tmp_mask = paddle.ones_like(mask,dtype=mask.dtype)
        # mask = paddle.cumsum(tmp_mask, axis=-1) == mask
        # mask = mask.scatter_(2, last_included, 1)

        # truncate unnecessary dims.
        max_dim = last_included.max()
        truncated_mask = mask[:, :, : max_dim + 1]
        truncated_probs = sorted_probs[:, :, : max_dim + 1]
        truncated_indices = sorted_indices[:, :, : max_dim + 1]

        # trim the words that are not in top-P by setting their probabilities
        # to 0, so that they would not be sampled later.
        trim_mask = ~truncated_mask
        trimed_probs = masked_fill(truncated_probs, trim_mask, 0)
        return trimed_probs, truncated_indices

    def step(
            self,
            step: int,
            lprobs,
            scores,
            prev_output_tokens=None,
            original_batch_idxs=None,
    ):

        bsz, beam_size, vocab_size = lprobs.shape

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :]

        if self.sampling_topp > 0:
            # only sample from the smallest set of words whose cumulative probability mass exceeds p
            probs, top_indices = self._sample_topp(lprobs)
        elif self.sampling_topk > 0:
            # only sample from top-k candidates
            lprobs, top_indices = lprobs.topk(self.sampling_topk)
            probs = paddle.exp(lprobs)
        else:
            probs = paddle.exp(lprobs)

            # dummy data to be consistent with true branch for type check
            top_indices = paddle.empty([0])
            # top_indices = paddle.cast(paddle.empty(0), dtype=probs.dtype)
        # sample
        if step == 0:
            indices_buf = paddle.multinomial(
                probs.reshape((bsz, -1)),
                beam_size,
                replacement=True,
            ).reshape((bsz, beam_size))
        else:
            indices_buf = paddle.multinomial(
                probs.reshape((bsz * beam_size, -1)),
                1,
                replacement=True,
            ).reshape((bsz, beam_size))

        if step == 0:
            # expand to beam size
            probs = probs.expand((bsz, beam_size, -1))

        # gather scores
        scores_buf = gather(probs, axis=2, index=indices_buf.unsqueeze(-1))
        scores_buf = paddle.log(scores_buf).reshape((bsz, -1))

        # remap indices if using top-k or top-P sampling
        if self.sampling_topk > 0 or self.sampling_topp > 0:
            indices_buf = gather(
                top_indices.expand((bsz, beam_size, -1)),
                axis=2,
                index=indices_buf.unsqueeze(-1),
            ).squeeze(2)

        if step == 0:
            beams_buf = paddle.zeros(shape=[bsz, beam_size], dtype=indices_buf.dtype)
        else:
            beams_buf = paddle.cast(paddle.arange(0, beam_size),dtype=indices_buf.dtype).tile((bsz, 1))
            # make scores cumulative
            scores_buf = scores_buf + gather(scores[:, :, step - 1], axis=1, index=beams_buf)
        return scores_buf, indices_buf, beams_buf


def masked_fill(x, mask, value):
    return paddle.where(mask, paddle.to_tensor(value, dtype=x.dtype), x)

def gather(x, axis, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if axis < 0:  # 最后一维
        axis = x.ndim + axis
    nd_index = []
    for k in range(x.ndim):
        if k == axis:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * x.ndim
            reshape_shape[k] = x.shape[k]
            dim_index = paddle.expand(paddle.arange(x.shape[k], dtype=index.dtype).reshape(reshape_shape),
                                      index_shape).flatten()
            nd_index.append(dim_index)
    paddle_out = paddle.gather_nd(x, paddle.stack(nd_index, axis=-1)).reshape(index_shape)
    return paddle_out

def scatter(tensor,dim,index,src):
    assert dim==0 or dim==1
    assert tensor.ndim==index.ndim==src.ndim==2
    index=paddle.cast(index,dtype='int64')
    i, j = index.shape
    grid_x, grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))
    if dim==0:
        index = paddle.stack([index.flatten(), grid_y.flatten()], axis=1)
    else:
        index = paddle.stack([grid_x.flatten(), index.flatten()], axis=1)
    # PaddlePaddle updates 的 shape 大小必须与 index 对应
    updates_index = paddle.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    updates = paddle.gather_nd(src, index=updates_index)
    res=paddle.scatter_nd_add(tensor, index, updates)
    return res
