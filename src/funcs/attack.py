import zlib

import torch

from tqdm import tqdm

from src.kvcache import (
    PICacheManager,
    cache_slice,
)
from src.kvcache.recomps import cache_recomputation
from src.utils.llm import build_kv_cache, greedy_decode


def uniform_chunks(total, max_size):
    """Split `total` rows into equally shaped chunks of at most `max_size`.

    Reduction order in fused GEMM/SDPA kernels depends on the batch shape, so in
    BF16 the same candidate scores differently in differently shaped microbatches
    (measured up to 0.37 nats against a candidate spread of ~3). Ranking is only
    self-consistent when every microbatch of a step has one shape.
    """
    if total < 1 or max_size < 1:
        raise ValueError('uniform_chunks requires positive total and max_size.')
    num_chunks = -(-total // max_size)
    return -(-total // num_chunks), num_chunks


class HijackKV:

    def __init__(self, model, tokenizer, device, args):
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.device = device
        self.args = args
        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = self._get_nonascii_ids()
        self.random_positions = None
        self.eval_batch_size = getattr(args, "eval_batch_size", 128)
        self.seed = getattr(args, "seed", 0)
        self.early_stop_patience = getattr(args, "early_stop_patience", 0)
        self.keep_best = getattr(args, "gcg_keep_best", False)
        self.generator = torch.Generator(device=self.device)
        self.topk = min(getattr(args, 'topk', 512), self.embedding_layer.num_embeddings - self.not_allowed_ids.numel())
        if self.topk < 1:
            raise ValueError('Tokenizer has no printable non-special candidate tokens.')
        if self.early_stop_patience < 0:
            raise ValueError('early_stop_patience must be non-negative.')

    def setup(self, picm):
        with torch.no_grad():
            self.cache_embeds = self.embedding_layer(picm.ids_group['cache'])
        self.query_ids = picm.ids_group['query']
        self.target_ids = picm.ids_group['target']
        if self.query_ids.shape[1] == 0 or self.target_ids.shape[1] == 0:
            raise ValueError('GCG requires non-empty query and target tokens.')
        self.input_ids = torch.cat([self.query_ids, self.target_ids[:, :-1]], dim=1)
        self.context_cache = picm.cache_group['context'] 
        self.benign_cache = picm.cache_group['benign'] 
        
        self.picm = picm
        # Seed per sample, not per worker: sharding samples over a different
        # number of GPUs must not change any sample's optimization trajectory.
        sample = getattr(picm, 'sample', None)
        sample_id = sample.get('id') if isinstance(sample, dict) else None
        stream = zlib.crc32(str(sample_id).encode('utf-8'))
        self.generator.manual_seed((int(self.seed) * 1_000_003 + stream) % (2 ** 63))

    def sample_random_positions(self, length, ratio):
        """Draw the shared Random-recomputation positions for one step."""
        count = int(length * ratio)
        if count <= 0 or count >= length:
            return None
        return torch.randperm(length, device=self.device, generator=self.generator)[:count].sort().values

    def run(self):
        prefix_ids = self._init_prefix_ids()
        if self.args.num_steps < 1 or self.args.eval_batch_size < 1 or self.args.search_width < 1:
            raise ValueError('num_steps, eval_batch_size and search_width must be positive.')
        best_step_idx = 0
        best_loss = float('inf')
        best_prefix_ids = prefix_ids.clone()
        selected_topk_ranks = []
        loss_curve = []
        hit_step = None
        hit_curve = []
        success_streak = 0
        steps_run = 0
        use_tqdm = len(getattr(self.args, "device", [])) <= 1

        for step_idx in tqdm(range(self.args.num_steps), dynamic_ncols=True, disable=not use_tqdm):
            steps_run = step_idx + 1
            if self.args.gcg_recomp_method == 'random':
                # Common random positions for the gradient and every candidate
                # microbatch in this step make candidate comparisons meaningful.
                self.random_positions = self.sample_random_positions(
                    self.cache_embeds.shape[1], self.args.gcg_recomp_ratio,
                )
            is_last_step = (step_idx == self.args.num_steps - 1)

            loss, grad, hit = self.compute_loss_and_grad(
                prefix_ids=prefix_ids, 
                required_grad=not is_last_step, 
                return_hit=True,
            )

            loss = loss.mean().item()
            loss_curve.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_prefix_ids = prefix_ids.clone()
                best_step_idx = step_idx

            step_hit = bool(hit.all().item())
            hit_curve.append(step_hit)
            if step_hit:
                if hit_step is None:
                    hit_step = step_idx
                success_streak += 1
            else:
                success_streak = 0
            # The target is already the greedy continuation; further steps only
            # sharpen a margin the decoder never reads.
            if self.early_stop_patience and success_streak >= self.early_stop_patience:
                break

            if not is_last_step:
                sampled_ids, topk_ids = self.sample_ids_from_grad(
                    grad=grad.squeeze(0),
                    prefix_ids=prefix_ids.squeeze(0),
                )

                prefix_ids, step_topk_ranks = self.batch_loss_eval(
                    sampled_ids=sampled_ids,
                    prefix_ids=prefix_ids,
                    topk_ids=topk_ids,
                )
                selected_topk_ranks.extend(step_topk_ranks)

        return dict(
            best_prefix_ids=best_prefix_ids,
            best_loss=best_loss,
            best_step=best_step_idx,
            selected_topk_ranks=selected_topk_ranks,
            loss_curve=loss_curve,
            hit_curve=hit_curve,
            steps_run=steps_run,
            hit_step=hit_step,
        )

    def compute_loss_and_grad(self, prefix_ids, required_grad=False, return_hit=False):
        with torch.set_grad_enabled(required_grad):
            batch_size = prefix_ids.shape[0]
            
            malicious_cache, prefix_embeds = self.build_malicious_cache(
                prefix_ids=prefix_ids,
                required_grad=required_grad,
            )
            
            input_cache = cache_recomputation(
                model=self.model,
                malicious_cache=malicious_cache,
                picm=self.picm, 
                ratio=self.args.gcg_recomp_ratio,
                method=self.args.gcg_recomp_method,
                include_context=True,
                random_positions=self.random_positions,
            )
            
            input_ids = self.input_ids.expand(batch_size, -1)
            shift = self.query_ids.shape[1]

            outputs = self.model.model(
                input_ids=input_ids,
                past_key_values=input_cache,
                use_cache=False,
                return_dict=True,
            )

            relevant_hidden = outputs.last_hidden_state[:, shift - 1 :, :]
            shift_logits = self.model.lm_head(relevant_hidden)

            shift_labels = self.target_ids.expand(batch_size, -1).to(self.device)
            token_loss = torch.nn.functional.cross_entropy(
                shift_logits.float().reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction='none',
            )
            loss = token_loss.view(batch_size, -1).mean(dim=1)

            hit = None
            if return_hit:
                # Greedy decoding already emits the target: the attack succeeded
                # under this recomputation, whatever the remaining loss margin.
                with torch.no_grad():
                    hit = shift_logits.argmax(dim=-1).eq(shift_labels).all(dim=1)

            grad = None
            if required_grad:
                prefix_embed_grads = None
                if loss.requires_grad:
                    prefix_embed_grads = torch.autograd.grad(
                        loss.mean(), prefix_embeds, allow_unused=True,
                    )[0]
                if prefix_embed_grads is None:
                    # Full recomputation removes every dependency on the prefix.
                    prefix_embed_grads = torch.zeros_like(prefix_embeds)
                grad = prefix_embed_grads @ self.embedding_layer.weight.T

        if return_hit:
            return loss, grad, hit
        return loss, grad

    def sample_ids_from_grad(self, grad, prefix_ids):
        with torch.no_grad():
            n_optim_tokens = len(prefix_ids)
            sampled_prefix_ids = prefix_ids.expand(self.args.search_width, -1).clone()

            # Low-precision backward can emit NaN/Inf; +inf is never picked by the
            # smallest-first top-k, so such entries drop out instead of poisoning it.
            finite = torch.isfinite(grad)
            if not finite.all():
                if not finite.any():
                    raise RuntimeError('Prefix gradient contains no finite entries.')
                grad = grad.masked_fill(~finite, float('inf'))

            if self.not_allowed_ids is not None:
                grad = grad.index_fill(1, self.not_allowed_ids, float("inf"))

            topk_ids = grad.topk(self.topk, dim=1, largest=False).indices
            
            sampled_ids_pos = torch.randint(
                0, n_optim_tokens,
                (self.args.search_width, self.args.n_replace),
                device=self.device, generator=self.generator,
            )
            
            random_topk_idx = torch.randint(
                0, self.topk,
                (self.args.search_width, self.args.n_replace),
                device=self.device, generator=self.generator,
            )
            sampled_ids_val = topk_ids[sampled_ids_pos, random_topk_idx]

            return sampled_prefix_ids.scatter_(1, sampled_ids_pos, sampled_ids_val), topk_ids

    def evaluate_candidates(self, candidates):
        """Score every candidate under one common microbatch shape.

        A CUDA OOM restarts the whole set at the smaller shape instead of
        continuing, because losses from two shapes cannot be ranked against
        each other. Padding rows are evaluated and discarded.
        """
        total = candidates.shape[0]
        while True:
            chunk, num_chunks = uniform_chunks(total, self.eval_batch_size)
            try:
                loss = []
                for start in range(0, num_chunks * chunk, chunk):
                    rows = candidates[start:start + chunk]
                    if rows.shape[0] < chunk:
                        rows = torch.cat([rows, rows[-1:].expand(chunk - rows.shape[0], -1)], dim=0)
                    loss.append(self.compute_loss_and_grad(prefix_ids=rows)[0])
                return torch.cat(loss, dim=0)[:total]
            except torch.cuda.OutOfMemoryError:
                if self.eval_batch_size <= 1:
                    raise
                self.eval_batch_size = max(1, self.eval_batch_size // 2)
                # No sample-level cache is mutated; retry the same candidates.
                torch.cuda.empty_cache()

    def batch_loss_eval(self, sampled_ids, prefix_ids, topk_ids):
        with torch.no_grad():
            candidates = sampled_ids
            if self.keep_best:
                # Scoring the incumbent inside the candidate batch is the only way
                # to compare it with the candidates at this precision, and it stops
                # the step from moving to a worse prefix. Measured on 20 HotPotQA
                # samples this smooths the search but slightly raises the final
                # loss, so plain GCG stays the default.
                candidates = torch.cat([sampled_ids, prefix_ids], dim=0)

            loss = self.evaluate_candidates(candidates)
            best_idx = loss.argmin()
            best_prefix_ids = candidates.index_select(0, best_idx.unsqueeze(0))
            selected_topk_ranks = self._selected_topk_ranks(
                prefix_ids=prefix_ids.squeeze(0),
                best_prefix_ids=best_prefix_ids.squeeze(0),
                topk_ids=topk_ids,
            )
            return best_prefix_ids, selected_topk_ranks

    def _init_prefix_ids(self):
        pad_token_id = self.tokenizer.encode("!!!!", add_special_tokens=False)
        prefix_ids = torch.full(
            (1, self.args.prefix_length), pad_token_id[0],
            dtype=torch.long, device=self.device,
        )
        return prefix_ids

    def build_malicious_cache(self, prefix_ids, required_grad=False):
        batch_size = prefix_ids.shape[0]

        with torch.set_grad_enabled(required_grad):
            prefix_embeds = self.embedding_layer(prefix_ids)
            if required_grad:
                prefix_embeds = prefix_embeds.detach().requires_grad_(True)

            cache_embeds = self.cache_embeds.expand(batch_size, -1, -1)
            malicious_embeds = torch.cat([prefix_embeds, cache_embeds], dim=1)

            full_malicious_cache = build_kv_cache(
                model=self.model,
                input_embeds=malicious_embeds,
                required_grad=required_grad,
            )
            malicious_cache = cache_slice(full_malicious_cache, start=prefix_ids.shape[1])

        return malicious_cache, prefix_embeds

    def _selected_topk_ranks(self, prefix_ids, best_prefix_ids, topk_ids):
        changed_positions = (best_prefix_ids != prefix_ids).nonzero(as_tuple=False).squeeze(-1)
        if changed_positions.numel() == 0:
            return []

        candidate_tokens = best_prefix_ids.index_select(0, changed_positions)
        candidate_topk = topk_ids.index_select(0, changed_positions)
        matches = candidate_topk.eq(candidate_tokens.unsqueeze(1))
        if not matches.any(dim=1).all():
            raise RuntimeError("Selected token is not contained in the recorded top-k candidates.")

        return matches.to(torch.int64).argmax(dim=1).add_(1).tolist()

    def _get_nonascii_ids(self, chunk_size=4096):
        vocab_size = min(len(self.tokenizer), self.embedding_layer.num_embeddings)
        nonascii_ids = set()

        for start_idx in range(0, vocab_size, chunk_size):
            end_idx = min(start_idx + chunk_size, vocab_size)
            ids = list(range(start_idx, end_idx))
            chunk_decoded_strings = self.tokenizer.batch_decode(ids, skip_special_tokens=False)

            for i, s in enumerate(chunk_decoded_strings):
                if not (s.isascii() and s.isprintable()):
                    nonascii_ids.add(start_idx + i)

        if hasattr(self.tokenizer, "all_special_ids"):
            nonascii_ids.update(i for i in self.tokenizer.all_special_ids if i < vocab_size)
        nonascii_ids.update(range(vocab_size, self.embedding_layer.num_embeddings))

        return torch.tensor(sorted(list(nonascii_ids)), device=self.device, dtype=torch.long)

def run_attack_sample(attacker, sample, args):
    tokenizer = attacker.tokenizer
    model = attacker.model
    device = attacker.device
    
    picm = PICacheManager(sample, model, tokenizer, device, args)

    benign_answer = greedy_decode(
        model=model,
        tokenizer=tokenizer,
        input_ids=picm.ids_group['query'],
        cache=picm.cache_group['full'],
        args=args,
    )
    
    attacker.setup(picm)
    result = attacker.run()

    selected_topk_ranks = result['selected_topk_ranks']
    if selected_topk_ranks:
        selected_topk_tensor = torch.tensor(selected_topk_ranks, dtype=torch.float32)
        selected_topk_mean = selected_topk_tensor.mean().item()
        selected_topk_var = selected_topk_tensor.var(unbiased=False).item()
    else:
        selected_topk_mean = None
        selected_topk_var = None

    return {
        "benign_answer": benign_answer,
        "loss": result['best_loss'],
        "best_step": result['best_step'],
        "best_ids": result['best_prefix_ids'],
        "steps_run": result['steps_run'],
        "hit_step": result['hit_step'],
        "selected_topk_mean": selected_topk_mean,
        "selected_topk_var": selected_topk_var,
        "loss_curve": result['loss_curve'],
        "hit_curve": result['hit_curve'],
    }
