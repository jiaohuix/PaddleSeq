import sys
import math
import paddle
import paddle.nn as nn
import numpy as np
from paddle import Tensor
import paddle.nn.functional as F
from .search import BeamSearch,Sampling


class SequenceGenerator(nn.Layer):
    def __init__(
            self,
            model,
            vocab_size,
            beam_size=1,
            max_len_a=0,
            max_len_b=200,
            max_len=0,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            # temperature=1.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=None,
            symbols_to_strip_from_output=None,
            need_attn=False,
    ):
        """Generates translations of a given source sentence.

        Args:
            model (List[~fairseq.model.FairseqModel]): ensemble of model,
                currently support fairseq.model.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            max_len (int, optional): the maximum length of the generated output
                (not including end-of-sentence)
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        self.model = model
        self.pad = model.pad_id
        self.unk = model.unk_id
        self.eos = model.eos_id
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b  # 200
        self.min_len = min_len  # 1
        # self.max_len = max_len or self.model.max_decoder_positions() # 1022
        self.max_len = 1022

        self.normalize_scores = normalize_scores  # true
        self.len_penalty = len_penalty  # 1
        self.unk_penalty = unk_penalty  # 0
        self.temperature = temperature  # 1
        self.match_source_len = match_source_len  # false
        self.repeat_ngram_blocker = None
        self.need_attn = need_attn
        assert temperature > 0, "--temperature must be greater than 0"

        # self.search = (
        #     BeamSearch() if search_strategy is None else search_strategy()
        # )
        self.search=BeamSearch() if search_strategy=="beam" else Sampling()
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread

        self.model.eval()

    def forward(
            self,
            sample,
            prefix_tokens=None,
            bos_token=None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (paddle.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    def _forward_decoder(self,
                         tokens,
                         encoder_out,
                         incremental_state,
                         temperature=1.0):
        logits, attn, incremental_state = self.model.forward_decoder(
            tokens,
            encoder_out,
            incremental_state,
        )

        def get_normalized_probs(x):
            return F.log_softmax(x, axis=-1, dtype='float32')
        if attn is not None:
            attn = attn[:, -1, :]
        lprobs = get_normalized_probs(logits / temperature)[:, -1, :]  # [bsz,vocab_size]
        return lprobs, attn, incremental_state

    @paddle.no_grad()
    def generate(self, sample, **kwargs):
        """Generate translations. Match the api of other fairseq generators.

        Args:
            model (List[~fairseq.model.FairseqModel]): ensemble of model
            sample (dict): batch
            prefix_tokens (paddle.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (paddle.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    def _generate(
            self,
            sample,
            prefix_tokens=None,
            constraints=None,
            bos_token=None,
    ):
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]  # [bsz src_len]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = paddle.cast(
                (src_tokens != self.eos) & (src_tokens != self.pad), dtype='int64'
            ).sum(axis=1)  # [bsz]
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].shape[-1] - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else paddle.to_tensor(src_tokens.shape[-1], dtype=src_tokens.dtype)
            )
        elif "features" in net_input:
            src_tokens = net_input["features"]
            src_lengths = (
                net_input["padding_mask"].shape[-1] - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else paddle.to_tensor(src_tokens.shape[-1], dtype=src_tokens.dtype)
            )
        else:
            raise Exception("expected src_tokens or source in net input. input keys: " + str(net_input.keys()))

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        if len(src_tokens.shape)==2:
            bsz, src_len = src_tokens.shape[:2]
        else: # for stream file
            bsz, src_len = src_tokens.shape[-2:]

        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        # self.search.init_constraints(constraints, beam_size) # pass

        max_len: int = -1
        if self.match_source_len:  # false,是否和src相同
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )  # 1022和200里选最小
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        encoder_out = self.model.forward_encoder(src_tokens)  # encoder输出是{},有多个模型，用[]包起来
        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores # 聚集分数？
        new_order = paddle.arange(bsz).reshape((-1, 1)).tile(repeat_times=[1, beam_size]).reshape([-1])  # [0 0 0 0 0 1 1 1 1 1 2 ... ...bsz-1] bsz*beam
        encoder_out = self.model.encoder.reorder_encoder_out(encoder_out, new_order)
        # ensure encoder_out is a List.
        assert encoder_out is not None  # [bsz*beam len dim]

        # initialize buffers
        incremental_state = self.model.decoder.gen_caches(encoder_out['encoder_out'][0])
        # [bsz*beam,max_len+1] 1是给eos
        scores = paddle.zeros((bsz * beam_size, max_len + 1))  # +1 for eos; pad is never chosen for scoring
        # tokens[bsz*beam max_len+2],pad为1，全填充
        # 所以tokens: bos+200+eos 202,而score是记录不确定的200+eos为201
        tokens = paddle.full(shape=[bsz * beam_size, max_len + 2], fill_value=self.pad, dtype=src_tokens.dtype)
        # bos_token=0
        tokens[:, 0] = self.eos if bos_token is None else bos_token  # 初始化为eos
        attn = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        # [bsz,beam],忽视多少个候选样本，假设已经2/5个完成了，盖掉那两个sample，然后只需要确定剩下的样本
        cands_to_ignore = paddle.zeros((bsz, beam_size)).equal(-1)  # forward and backward-compatible False mask

        # list of completed sentences [bsz]
        finalized = [[] for i in range(bsz)]
        # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # 每个step保留2beam个候选，防止一大半eos早停
        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # bbsz_offsets[0 5 10 ... (bsz-1)*beam]
        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (paddle.arange(0, bsz) * beam_size)
                .unsqueeze(1)
        )
        # [0,1,...,2beam-1]
        cand_offsets = paddle.arange(0, cand_size)

        reorder_state = None
        batch_idxs = None

        original_batch_idxs = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = paddle.arange(0, bsz)

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - paddle.arange(batch_idxs.numel(), dtype=batch_idxs.dtype)
                    reorder_state = (reorder_state.reshape([-1, beam_size]) + corr.unsqueeze(-1) * beam_size).reshape(
                        [-1])
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                incremental_state = self.model.decoder.reorder_incremental_state(incremental_state, reorder_state)
                encoder_out = self.model.encoder.reorder_encoder_out(
                    encoder_out, reorder_state
                )


            lprobs, avg_attn_scores, incremental_state = self._forward_decoder(
                tokens[:, : step + 1],
                encoder_out,
                incremental_state,
                self.temperature)
            avg_attn_scores = avg_attn_scores if self.need_attn else None
            # [bsz vocab_size]
            lprobs[lprobs != lprobs] = paddle.to_tensor(-math.inf, dtype=lprobs.dtype)
            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty unk3,pen0

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                    prefix_tokens is not None
                    and step < prefix_tokens.shape[1]
                    and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:  # none
                if attn is None:
                    attn = paddle.empty(
                        (bsz * beam_size, avg_attn_scores.shape[1], max_len + 2))
                attn[:, :, step + 1] = avg_attn_scores

            scores = paddle.cast(scores, dtype=lprobs.dtype)
            eos_bbsz_idx = paddle.empty([0], dtype=tokens.dtype)
            # indices of hypothesis ending with eos (finished sentences)
            eos_scores = paddle.empty([0])  # scores of hypothesis ending with eos (finished sentences)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.reshape((bsz, -1, self.vocab_size)),
                scores.reshape((bsz, beam_size, -1))[:, :, :step] if step != 0 else None,
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams + bbsz_offsets

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.equal(self.eos) & cand_scores.not_equal(paddle.to_tensor(-math.inf))
            eos_mask = paddle.cast(eos_mask, dtype='float32')
            eos_mask[:, :beam_size][cands_to_ignore] = 0. 
            eos_mask = paddle.cast(eos_mask, dtype='bool')
            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = paddle.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = paddle.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )
                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)
                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = paddle.ones([bsz, ], dtype=paddle.bool)
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after paddleScript supports it
                batch_idxs = paddle.arange(
                    bsz
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                # self.search.prune_sentences(batch_idxs)
                eos_mask = paddle.cast(paddle.cast(eos_mask, dtype='int64')[batch_idxs], dtype='bool')
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets = bbsz_offsets[:new_bsz, :]
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = paddle.cast(paddle.cast(cands_to_ignore, dtype='int64')[batch_idxs], dtype='bool')
                scores = scores.reshape((bsz, -1))[batch_idxs].reshape((new_bsz * beam_size, -1))
                tokens = tokens.reshape((bsz, -1))[batch_idxs].reshape((new_bsz * beam_size, -1))
                if attn is not None:
                    attn = attn.reshape((bsz, -1))[batch_idxs].reshape(
                        [new_bsz * beam_size, attn.shape[1], -1]
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in paddlescript.
            # assert len(eos_mask.shape) == len(cands_to_ignore.shape)
            if len(eos_mask.shape) < 2:  # index_sample只能用于2d，不能完全替代gather
                if isinstance(self.search,BeamSearch):
                    eos_mask, cands_to_ignore = eos_mask.reshape([-1, cand_size]), cands_to_ignore.reshape([-1, beam_size])  #[bsz,cand_size=2*beam] [bsz,beam]
                else:
                    eos_mask, cands_to_ignore = eos_mask.reshape([-1, beam_size]), cands_to_ignore.reshape([-1, beam_size])  #[bsz,cand_size=2*beam] [bsz,beam]
                cand_scores = cand_scores.reshape([1, -1])
                cand_indices = cand_indices.reshape([1, -1])

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = paddle.cast(eos_mask, dtype=cand_offsets.dtype) * cand_size + cand_offsets[
                                                                                        : eos_mask.shape[1]]

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = paddle.topk(
                active_mask, k=beam_size, axis=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = (new_cands_to_ignore >= cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(axis=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            # active_bbsz_idx = paddle.index_sample(cand_bbsz_idx, index=active_hypos)
            # active_scores = paddle.index_sample(cand_scores, index=active_hypos)
            active_bbsz_idx = gather(cand_bbsz_idx, axis=1, index=active_hypos)
            active_scores = gather(cand_scores, axis=1, index=active_hypos)
            active_bbsz_idx = active_bbsz_idx.reshape([-1])
            active_scores = active_scores.reshape([-1])

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = paddle.index_select(
                tokens[:, : step + 1], axis=0, index=active_bbsz_idx
            )

            # Select the next token for each of them
            # tokens [bsz*beam,202] cand_indices=[bsz,5](after gather)
            # tokens[:, step + 1] = paddle.index_sample(cand_indices, index=active_hypos).reshape([-1])
            tokens[:, step + 1] = gather(cand_indices,axis=1,index=active_hypos).reshape([-1])
            if step > 0:
                scores[:, :step] = paddle.index_select(
                    scores[:, :step], axis=0, index=active_bbsz_idx
                )
            # scores[:, step] = paddle.index_sample(cand_scores, index=active_hypos).reshape([-1])
            scores[:, step] = gather(cand_scores,axis=1,index=active_hypos).reshape([-1])

            # Update constraints based on which candidates were selected for the next beam
            # self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = paddle.index_select(
                    attn[:, :, : step + 2], axis=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = paddle.to_tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            # _, sorted_scores_indices = paddle.sort(scores, descending=True) # values,indices
            sorted_scores_indices = paddle.argsort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = finalized[sent]

        return finalized

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.reshape([-1, beam_size, tensor.shape[-1]])
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.reshape([-1, tensor.shape[-1]])

    def finalize_hypos(
            self,
            step: int,
            bbsz_idx,
            eos_scores,
            tokens,
            scores,
            finalized,
            finished,
            beam_size,
            attn,
            src_lengths,
            max_len,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(axis=0, index=bbsz_idx)[
                       :, 1: step + 2
                       ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(axis=0, index=bbsz_idx)[:, :, 1: step + 2]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(axis=0, index=bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = paddle.to_tensor(cum_unfin, dtype=bbsz_idx.dtype)
        unfin_idx = bbsz_idx // beam_size
        sent = unfin_idx + paddle.index_select(cum_fin_tensor, axis=0, index=unfin_idx)

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent.numpy() << 32) + unfin_idx.numpy()
        unique_seen = np.unique(seen).tolist()

        if self.match_source_len:
            condition = step > paddle.index_select(src_lengths, axis=0, index=unfin_idx)
            eos_scores = paddle.where(condition, paddle.to_tensor(-math.inf), eos_scores)
        sent_list = sent.tolist()
        for i in range(bbsz_idx.shape[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = paddle.empty([0])

                finalized[sent_list[i]].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": eos_scores[i],
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": paddle.empty([0]),
                        "positional_scores": pos_scores[i],
                    }
                )

        newly_finished = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)
            # unique_sent出界了
            if not finished[unique_sent] and self.is_finished(
                    step, unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size
            ):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished

    def is_finished(
            self,
            step: int,
            unfin_idx: int,
            max_len: int,
            finalized_sent_len: int,
            beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False


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

