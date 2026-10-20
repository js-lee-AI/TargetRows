"""Optional backend checks using tiny random models; no downloads or GPU needed."""
import importlib.util
import unittest


def _backend_installed():
    if not all(importlib.util.find_spec(name) for name in ("torch", "transformers")):
        return False
    import transformers
    return transformers.__version__.split(".")[0] == "4"


HAS_BACKEND = _backend_installed()
if HAS_BACKEND:
    import numpy as np
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
    from target_rows.decoding import ar_decode, decode, draft_pool_fast, tree_block
    from target_rows.drafter import LocalDrafter
    from target_rows.torch_backend import FastLM


@unittest.skipUnless(HAS_BACKEND, "Install the gpu extra (Transformers 4.x) to run backend checks.")
class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def models(self, kind="qwen3", seed=11):
        torch.manual_seed(seed)
        config_class, model_class = ((Qwen3Config, Qwen3ForCausalLM) if kind == "qwen3"
                                    else (LlamaConfig, LlamaForCausalLM))
        config = config_class(vocab_size=32, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=8, max_position_embeddings=512,
                              attention_dropout=0.0)
        config._attn_implementation = "eager"
        hf = model_class(config).eval()
        return hf, FastLM.from_model(hf, max_len=256)

    def hf_row(self, hf, tokens):
        with torch.no_grad():
            return hf(torch.tensor([tokens])).logits[0, -1]

    def test_prefill_tree_rows_and_cache_commit_match_hugging_face(self):
        for kind in ("llama", "qwen3"):
            with self.subTest(kind=kind):
                hf, lm = self.models(kind)
                prompt = [1, 2, 3]
                torch.testing.assert_close(lm.prefill(prompt), self.hf_row(hf, prompt), atol=2e-6, rtol=2e-5)
                # Root 4, branches 5 -> 7 -> 8 and 6.
                parents = np.array([0, 0, 1, 3])
                block = tree_block(parents, with_root=True)
                ids = torch.tensor([4, 5, 6, 7, 8])
                positions = block.sum(1) - 1 + len(prompt)
                slots = torch.arange(len(prompt), len(prompt) + len(ids))

                def fill(mask):
                    mask[:, :len(prompt)] = True
                    mask[:, len(prompt):len(prompt) + len(ids)] = block

                logits = lm.run(ids, positions, slots, fill)
                paths = [[4], [4, 5], [4, 6], [4, 5, 7], [4, 5, 7, 8]]
                for row, path in zip(logits, paths):
                    torch.testing.assert_close(row, self.hf_row(hf, prompt + path), atol=2e-6, rtol=2e-5)
                accepted = torch.tensor([3, 4, 6, 7])
                lm.k[:, :, 3:7] = lm.k[:, :, accepted]
                lm.v[:, :, 3:7] = lm.v[:, :, accepted]
                one = torch.tensor([7])
                following = lm.run(torch.tensor([9]), one, one, lambda mask: mask[:, :8].fill_(True))[0]
                torch.testing.assert_close(following, self.hf_row(hf, prompt + paths[-1] + [9]), atol=2e-6, rtol=2e-5)

    def test_draft_pool_matches_hugging_face_path_scores(self):
        hf, lm = self.models()
        prompt = [1, 2, 3]
        lm.prefill(prompt)
        tokens, scores, parents, n_done = draft_pool_fast(lm, [4], len(prompt), 32)
        self.assertEqual(len(tokens), 710)
        self.assertEqual(n_done, 4)
        paths = []
        expected = []
        for i, (token, parent) in enumerate(zip(tokens, parents)):
            self.assertLess(parent, i)
            prefix = [] if parent < 0 else paths[parent]
            paths.append(prefix + [int(token)])
            expected.append(float(torch.log_softmax(self.hf_row(hf, prompt + [4] + prefix), -1)[token])
                            + (0 if parent < 0 else expected[parent]))
        np.testing.assert_allclose(scores, expected, atol=3e-6, rtol=2e-6)

    def test_every_policy_matches_greedy_ar_across_resets(self):
        for kind in ("llama", "qwen3"):
            _, target = self.models(kind, seed=11)
            _, draft = self.models(kind, seed=29)
            drafter = LocalDrafter(draft, 32)
            for prompt in ([1, 2, 3], [7, 2]):
                reference, _ = ar_decode(target, prompt, set(), 12)
                for policy in ("draft", "root", "root-draw", "preselected"):
                    with self.subTest(kind=kind, policy=policy, prompt=prompt):
                        result = decode(target, drafter, prompt, 0.0, 8, policy, 5, set(), 12)
                        self.assertEqual(result["emitted"], reference)
                        self.assertEqual(result["tokens"], 12)

    def test_first_eos_and_one_token_limit(self):
        _, target = self.models()
        _, draft = self.models(seed=29)
        drafter = LocalDrafter(draft, 32)
        first = int(target.prefill([1, 2, 3]).argmax())
        for policy in ("draft", "root", "root-draw", "preselected"):
            for eos, limit in (({first}, 12), (set(), 1)):
                result = decode(target, drafter, [1, 2, 3], 0.0, 8, policy, 5, eos, limit)
                self.assertEqual(result["emitted"], [first])
                self.assertEqual(result["rounds"], 0)

    def test_sampling_repeats_with_the_same_seed_after_reset(self):
        _, target = self.models()
        _, draft = self.models(seed=29)
        drafter = LocalDrafter(draft, 32)
        for policy in ("draft", "root", "root-draw", "preselected"):
            with self.subTest(policy=policy):
                first = decode(target, drafter, [1, 2, 3], 1.0, 8, policy, 17, set(), 12)
                repeat = decode(target, drafter, [1, 2, 3], 1.0, 8, policy, 17, set(), 12)
                self.assertEqual(first["emitted"], repeat["emitted"])
                self.assertEqual(len(first["emitted"]), 12)
                self.assertTrue(all(0 <= token < 32 for token in first["emitted"]))

    def test_invalid_settings_rejected(self):
        _, target = self.models()
        _, draft = self.models(seed=29)
        drafter = LocalDrafter(draft, 32)
        for temperature, budget, count in ((-1, 8, 5), (float("nan"), 8, 5), (0, 0, 5), (0, 8, 0), (0, 8, 256)):
            with self.assertRaises(ValueError):
                decode(target, drafter, [1, 2, 3], temperature, budget, "root", 1, set(), count)


if __name__ == "__main__":
    unittest.main()
