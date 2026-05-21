import torch

from tqdm import tqdm

from src.kvcache import (
    PICacheManager,
    cache_split,
    cache_expand,
    cache_concat,
)
from src.kvcache.recomps import cache_recomputation
from src.utils.llm import build_kv_cache, greedy_decode

class HijackKV:

    def __init__(self, model, tokenizer, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.args = args
        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = self._get_nonascii_ids()
    
    def setup(self, picm):
        with torch.no_grad():
            self.cache_embeds = self.embedding_layer(picm.ids_group['cache'])
        self.query_ids = picm.ids_group['query']
        self.target_ids = picm.ids_group['target']
        self.input_ids = torch.cat([self.query_ids, self.target_ids], dim=1)
        self.context_cache = picm.cache_group['context'] 
        self.benign_cache = picm.cache_group['benign'] 
        
        self.picm = picm

    def run(self):
        prefix_ids = self._init_prefix_ids()
        best_loss = float('inf')
        best_prefix_ids = prefix_ids.clone()
        selected_topk_ranks = []
        loss_curve = []
        use_tqdm = len(getattr(self.args, "device", [])) <= 1

        for step_idx in tqdm(range(self.args.num_steps), dynamic_ncols=True, disable=not use_tqdm):
            is_last_step = (step_idx == self.args.num_steps - 1)

            loss, grad = self.compute_loss_and_grad(
                prefix_ids=prefix_ids, 
                required_grad=not is_last_step, 
            )

            loss = loss.mean().item()
            loss_curve.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_prefix_ids = prefix_ids.clone()
                best_step_idx = step_idx

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

        return best_prefix_ids, best_loss, best_step_idx, selected_topk_ranks, loss_curve

    def compute_loss_and_grad(self, prefix_ids, required_grad=False):
        with torch.set_grad_enabled(required_grad):
            batch_size = prefix_ids.shape[0]
            
            malicious_cache, prefix_embeds = self.build_malicious_cache(
                prefix_ids=prefix_ids,
                required_grad=required_grad,
            )
            
            recomp_cache = cache_recomputation(
                model=self.model,
                malicious_cache=malicious_cache,
                picm=self.picm, 
                ratio=self.args.gcg_recomp_ratio,
                method=self.args.gcg_recomp_method,
            )
            
            batch_context_cache = cache_expand(
                cache=self.context_cache,
                batch_size=batch_size,
            )
            input_cache = cache_concat(cache_list=[batch_context_cache, recomp_cache])

            input_ids = self.input_ids.expand(batch_size, -1)
            shift = self.query_ids.shape[1]

            outputs = self.model.model(
                input_ids=input_ids,
                past_key_values=input_cache,
                use_cache=False,
                return_dict=True,
            )

            relevant_hidden = outputs.last_hidden_state[:, shift - 1 : -1, :]
            shift_logits = self.model.lm_head(relevant_hidden)

            shift_labels = self.target_ids.expand(batch_size, -1).to(self.device)
            token_loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction='none',
            )
            loss = token_loss.view(batch_size, -1).mean(dim=1)

            grad = None
            if required_grad:
                prefix_embed_grads = torch.autograd.grad(
                    outputs=[loss.mean()],
                    inputs=[prefix_embeds],
                    retain_graph=False,
                    create_graph=False,
                )[0]
                grad = prefix_embed_grads @ self.embedding_layer.weight.T

        return loss, grad

    def sample_ids_from_grad(self, grad, prefix_ids):
        with torch.no_grad():
            n_optim_tokens = len(prefix_ids)
            sampled_prefix_ids = prefix_ids.expand(self.args.search_width, -1).clone()

            if self.not_allowed_ids is not None:
                grad.index_fill_(1, self.not_allowed_ids, float("inf"))

            topk_ids = grad.topk(self.args.topk, dim=1, largest=False).indices
            
            sampled_ids_pos = torch.randint(
                0, n_optim_tokens,
                (self.args.search_width, self.args.n_replace),
                device=self.device,
            )
            
            random_topk_idx = torch.randint(
                0, self.args.topk,
                (self.args.search_width, self.args.n_replace),
                device=self.device,
            )
            sampled_ids_val = topk_ids[sampled_ids_pos, random_topk_idx]

            return sampled_prefix_ids.scatter_(1, sampled_ids_pos, sampled_ids_val), topk_ids

    def batch_loss_eval(self, sampled_ids, prefix_ids, topk_ids):
        with torch.no_grad():
            loss = []
            num_candidates = sampled_ids.shape[0]

            for i in range(0, num_candidates, self.args.eval_batch_size):
                batch_end = min(i + self.args.eval_batch_size, num_candidates)
                batch_prefix_ids = sampled_ids[i:batch_end]

                _loss, _ = self.compute_loss_and_grad(prefix_ids=batch_prefix_ids)
                loss.append(_loss)

            loss = torch.cat(loss, dim=0)
            best_idx = loss.argmin()
            best_prefix_ids = sampled_ids.index_select(0, best_idx.unsqueeze(0))
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
            _, malicious_cache = cache_split(
                cache=full_malicious_cache,
                k=prefix_ids.shape[1],
            )

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
        vocab_size = self.tokenizer.vocab_size
        nonascii_ids = set()

        for start_idx in range(0, vocab_size, chunk_size):
            end_idx = min(start_idx + chunk_size, vocab_size)
            ids = list(range(start_idx, end_idx))
            chunk_decoded_strings = self.tokenizer.batch_decode(ids, skip_special_tokens=False)

            for i, s in enumerate(chunk_decoded_strings):
                if not (s.isascii() and s.isprintable()):
                    nonascii_ids.add(start_idx + i)

        if hasattr(self.tokenizer, "all_special_ids"):
            nonascii_ids.update(self.tokenizer.all_special_ids)

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
    best_prefix_ids, best_loss, best_step, selected_topk_ranks, loss_curve = attacker.run()
    
    if selected_topk_ranks:
        selected_topk_tensor = torch.tensor(selected_topk_ranks, dtype=torch.float32)
        selected_topk_mean = selected_topk_tensor.mean().item()
        selected_topk_var = selected_topk_tensor.var(unbiased=False).item()
    else:
        selected_topk_mean = None
        selected_topk_var = None

    return {
        "benign_answer": benign_answer,
        "loss": best_loss,
        "best_step": best_step,
        "best_ids": best_prefix_ids,
        "selected_topk_mean": selected_topk_mean,
        "selected_topk_var": selected_topk_var,
        "loss_curve": loss_curve,
    }
