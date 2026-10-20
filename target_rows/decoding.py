"""Standalone-drafter decoding with root, root-draw and draft tree selection."""
import time
import numpy as np
import torch

TOP_K, DEPTH = 10, 7
N_CAND = TOP_K + DEPTH * TOP_K * TOP_K
LEVEL = np.concatenate([np.ones(TOP_K, dtype=np.int64)] +
                       [np.full(TOP_K * TOP_K, 2 + i, dtype=np.int64) for i in range(DEPTH)])
NEG = -1e30


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def validate_decode(lm, prompt, temperature, budget, max_new):
    if not prompt or max_new < 1 or not 1 <= budget <= N_CAND:
        raise ValueError("A nonempty prompt, positive token limit, and budget in [1, 710] are required.")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative.")
    reserve = max(budget + 1, TOP_K * DEPTH) + DEPTH + 3
    if len(prompt) + max_new + reserve >= lm.max_len - 1:
        raise ValueError("Prompt, generation limit and tree scratch space exceed --max-context; increase it.")


def rule_order(score, level):
    """Rank candidates by cumulative draft score, then shallower first, then index."""
    n = score.shape[0]
    idx = np.arange(n)
    order = np.lexsort((idx, level, -score))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return order, rank


def first_ancestor(par):
    """First-level ancestor of every pool candidate."""
    anc = np.arange(N_CAND)
    for g in range(TOP_K, N_CAND):
        anc[g] = anc[par[g]]
    return anc


def select_root(score, rank, anc1, lp1_first):
    """Top-B order by p(x_1 | c) times the deeper draft factors."""
    w = lp1_first[anc1] + (score - score[anc1])
    return np.lexsort((rank, LEVEL, -w))


def subtree_keep(anc1, rank, g1, budget):
    """The drawn first-level candidate g1 and at most budget - 1 of its descendants in draft order
    for a fixed node budget."""
    desc = np.nonzero((anc1 == g1) & (LEVEL >= 2))[0]
    desc = desc[np.argsort(rank[desc], kind="stable")][: budget - 1]
    return np.concatenate([[g1], desc])



@torch.no_grad()
def draft_pool_fast(lm, new_tokens, n_done, n_vocab):
    """Write new_tokens (which end with the continuation token) at slots n_done.. and grow the pool.

    Returns (tokens, scores, parents) and the drafter's committed length n.
    Candidates 0..9 are the ten most probable tokens after the continuation token. In expansion round i (0..6) the
    ten frontier candidates are fed at position n + i into scratch slots n + 10 i .. n + 10 i + 9, each attending to
    the committed slots and to the slots of its own ancestors, and their ten most probable children become
    candidates 10 + 100 i + 10 j + k. The scratch slots are overwritten by the next round's committed tokens.
    """
    dev = lm.dev
    n = n_done + len(new_tokens)
    if not new_tokens or n + TOP_K * DEPTH >= lm.max_len - 1:
        raise ValueError("New tokens and draft scratch slots must fit in the cache.")
    cols = torch.arange(lm.max_len, device=dev)
    ar = torch.arange(n_done, n, device=dev)
    logits = lm.run(torch.tensor(new_tokens, dtype=torch.long, device=dev), ar, ar,
                    lambda m: m.copy_(cols[None, :] <= ar[:, None]))[-1].float()
    logits[n_vocab:] = float("-inf")
    top = torch.topk(torch.log_softmax(logits, -1), TOP_K)
    tokens, scores = [top.indices], [top.values]
    parents = [torch.full((TOP_K,), -1, dtype=torch.long, device=dev)]
    front_tok, front_score = top.indices, top.values
    front_idx = torch.arange(TOP_K, device=dev)
    anc = torch.eye(TOP_K, dtype=torch.bool, device=dev)  # frontier rows over the scratch slots written so far
    eye = torch.eye(TOP_K, dtype=torch.bool, device=dev)
    for i in range(DEPTH):
        slots = torch.arange(n + TOP_K * i, n + TOP_K * (i + 1), device=dev)
        pos = torch.full((TOP_K,), n + i, dtype=torch.long, device=dev)

        def fill(m, anc=anc):
            m[:, :n] = True
            m[:, n:n + anc.shape[1]] = anc

        lg = lm.run(front_tok, pos, slots, fill).float()
        lg[:, n_vocab:] = float("-inf")
        top = torch.topk(torch.log_softmax(lg, -1), TOP_K, dim=-1)
        cu = top.values + front_score[:, None]
        tokens.append(top.indices.reshape(-1))
        scores.append(cu.reshape(-1))
        parents.append(front_idx.repeat_interleave(TOP_K))
        sel = torch.topk(cu.reshape(-1), TOP_K)
        front_tok = top.indices.reshape(-1)[sel.indices]
        front_score = sel.values
        front_idx = TOP_K + TOP_K * TOP_K * i + sel.indices
        anc = torch.cat([anc[sel.indices // TOP_K], eye], dim=1)
    return (torch.cat(tokens).cpu().numpy().astype(np.int64), torch.cat(scores).float().cpu().numpy(),
            torch.cat(parents).cpu().numpy().astype(np.int64), n)


def sample(row, temp, gen):
    if temp == 0:
        return int(row.argmax())
    return int(torch.multinomial(torch.softmax(row / temp, -1), 1, generator=gen))


def tree_block(par_local, with_root):
    """Ancestor mask of the fed tokens (parents first). Local index 0 is c, which only the draft decoder feeds."""
    m = len(par_local)
    full = torch.eye(m + 1, dtype=torch.bool)
    if with_root:
        full[:, 0] = True
    for k, p in enumerate(par_local.tolist()):
        full[k + 1] |= full[p]
    return full if with_root else full[1:, 1:]


@torch.no_grad()
def decode(lm, drafter, prompt_ids, temp, B, policy, seed, eos, max_new, round_limit=False):
    """Decode with a fixed 710-node candidate pool and target sampling.

    Returned round statistics count the work actually performed. By default,
    output tokens stop at max_new or the first EOS, even if the last round
    computed a longer path. round_limit keeps the paper's after-round limit.
    """
    validate_decode(lm, prompt_ids, temp, B, max_new)
    if policy not in ("draft", "root", "root-draw", "preselected"):
        raise ValueError("Unknown tree selection policy.")
    dev = lm.dev
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    synchronize(dev)
    t_start = time.perf_counter()
    drafter.reset(prompt_ids)
    c = sample(lm.prefill(prompt_ids).float(), temp, gen)
    t_prefill = time.perf_counter() - t_start
    L = len(prompt_ids)  # slot and position of c, the continuation token whose row is not yet computed
    emitted = [c]
    to_drafter = [c]
    stop = c in eos
    rec = {k: [] for k in ("A", "size", "hit", "t_root", "t_wait", "t_sel", "t_tree", "t_walk", "t_draft", "t_round")}
    while not stop and len(emitted) < max_new:
        if L + max(B + 1, TOP_K * DEPTH) >= lm.max_len - 1:
            raise ValueError("Context and tree scratch slots exceed the cache.")
        tr0 = time.perf_counter()
        drafter.submit(to_drafter)
        root_row, x1 = None, None
        if policy != "draft":
            # root pass on c while the drafter drafts
            one = torch.tensor([L], device=dev)
            root_row = lm.run(torch.tensor([c], device=dev), one, one, lambda m: m[:, :L + 1].fill_(True))[0].float()
            x1 = sample(root_row, temp, gen)
        tr1 = time.perf_counter()
        tok, score, par, t_draft = drafter.result()
        tr2 = time.perf_counter()
        _, rank = rule_order(score.astype(np.float32), LEVEL)
        par0 = np.where(par < 0, 0, par)
        anc1 = first_ancestor(par0)
        special, skip, hit = None, False, 1
        if policy == "draft":
            keep = np.nonzero(rank < B)[0]
        elif policy == "root":
            if temp == 0:
                lp1 = np.where(tok[:TOP_K] == int(root_row.argmax()), 0.0, NEG)
            else:
                lsm = torch.log_softmax(root_row / temp, -1)
                lp1 = lsm[torch.from_numpy(tok[:TOP_K]).to(dev)].double().cpu().numpy()
            lp1_first = np.zeros(N_CAND)
            lp1_first[:TOP_K] = lp1
            keep = np.sort(select_root(score.astype(np.float64), rank, anc1, lp1_first)[:B])
            first = keep[keep < TOP_K]
            if not np.any(tok[first] == x1):
                skip, hit = True, 0
        else:
            g = np.nonzero(tok[:TOP_K] == x1)[0]
            if policy == "preselected":
                g = g[rank[g] < B]
            if len(g) and policy == "root-draw":
                keep = np.sort(subtree_keep(anc1, rank, int(g[0]), B))
            elif len(g):
                keep = np.sort(np.nonzero((anc1 == int(g[0])) & (rank < B))[0])
            else:
                keep = np.zeros(0, dtype=np.int64)
                special, hit = x1, 0
        eos_credit = 0
        if policy != "draft" and x1 in eos:
            # Count the pre-drawn token when it belongs to the selected tree.
            eos_credit = int(not skip)
            skip = True
        tr3 = time.perf_counter()
        if skip:
            # x1 is the round's continuation: c's row came from the root pass, whose slot L is already written
            path, toks, b = [], [c], x1
            tr4 = tr3
        else:
            if special is None:
                local = np.full(N_CAND, -1, dtype=np.int64)
                local[keep] = np.arange(1, len(keep) + 1)
                par_local = np.where(par[keep] < 0, 0, local[par0[keep]])
                if not ((par_local >= 0).all() and (par_local < np.arange(1, len(keep) + 1)).all()):
                    raise RuntimeError("Selected tree is not prefix-closed.")
                cand = tok[keep].tolist()
            else:
                par_local = np.zeros(1, dtype=np.int64)
                cand = [special]
            with_root = policy == "draft"
            blk = tree_block(par_local, with_root)
            depth = blk.sum(1)  # number of ancestors in the block, the token itself included
            toks = [c] + cand
            if with_root:
                feed, pos, slot0, ctx = toks, depth - 1 + L, L, L
            else:
                feed, pos, slot0, ctx = cand, depth + L, L + 1, L + 1
            nf = len(feed)
            blk_d = blk.to(dev)

            def fill(m):
                m[:, :ctx] = True
                m[:, slot0:slot0 + nf] = blk_d

            logits = lm.run(torch.tensor(feed, device=dev), pos.to(dev), torch.arange(slot0, slot0 + nf, device=dev),
                            fill)
            off = 0 if with_root else 1  # logits row of local index j is j - off
            synchronize(dev)
            tr4 = time.perf_counter()
            children = {}
            for j, p in enumerate(par_local.tolist()):
                children.setdefault(p, {})[cand[j]] = j + 1
            node, path = 0, []
            x = x1 if policy != "draft" else sample(logits[0].float(), temp, gen)
            while True:
                nxt = children.get(node, {}).get(x)
                if nxt is None:
                    b = x
                    break
                path.append(nxt)
                node = nxt
                x = sample(logits[node - off].float(), temp, gen)
            del logits
        # commit c and the accepted path: local index j sits in slot L + j
        if len(path):
            acc = torch.tensor([0] + path, device=dev) + L
            lm.k[:, :, L:L + len(path) + 1] = lm.k[:, :, acc]
            lm.v[:, :, L:L + len(path) + 1] = lm.v[:, :, acc]
        new = [toks[j] for j in path] + [b]
        emitted.extend(new)
        to_drafter = new
        L += len(path) + 1
        c = b
        if any(t in eos for t in new):
            stop = True
        tr5 = time.perf_counter()
        rec["A"].append(len(path) + eos_credit)
        rec["size"].append(0 if skip else len(toks) - 1)
        rec["hit"].append(hit)
        rec["t_root"].append(tr1 - tr0)
        rec["t_wait"].append(tr2 - tr1)
        rec["t_sel"].append(tr3 - tr2)
        rec["t_tree"].append(tr4 - tr3)
        rec["t_walk"].append(tr5 - tr4)
        rec["t_draft"].append(t_draft)
        rec["t_round"].append(tr5 - tr0)
    synchronize(dev)
    seconds = time.perf_counter() - t_start
    n_tok = len(emitted) if round_limit else min(len(emitted), max_new)
    for k, t in enumerate(emitted[:n_tok]):
        if t in eos:
            n_tok = k + 1
            break
    line = {"tokens": n_tok, "seconds": seconds, "prefill_seconds": t_prefill, "rounds": len(rec["A"]),
            "sum_A": int(sum(rec["A"])), "sum_size": int(sum(rec["size"])), "hits": int(sum(rec["hit"])),
            "emitted": emitted[:n_tok]}
    for k in ("t_root", "t_wait", "t_sel", "t_tree", "t_walk", "t_draft", "t_round"):
        v = np.array(rec[k]) if rec[k] else np.zeros(1)
        line[k + "_sum"] = float(v.sum())
        line[k + "_median"] = float(np.median(v))
    line["A"] = rec["A"]
    line["size"] = rec["size"]
    return line


@torch.no_grad()
def ar_decode(lm, prompt_ids, eos, max_new, temp=0.0, seed=0):
    """Autoregressive baseline using the same target backend and temperature."""
    validate_decode(lm, prompt_ids, temp, 1, max_new)
    dev = lm.dev
    synchronize(dev)
    t0 = time.perf_counter()
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = sample(lm.prefill(prompt_ids).float(), temp, gen)
    out, L = [x], len(prompt_ids)
    while len(out) < max_new and x not in eos:
        one = torch.tensor([L], device=dev)
        row = lm.run(torch.tensor([x], device=dev), one, one, lambda m: m[:, :L + 1].fill_(True))[0].float()
        x = sample(row, temp, gen)
        out.append(x)
        L += 1
    synchronize(dev)
    return out, time.perf_counter() - t0
