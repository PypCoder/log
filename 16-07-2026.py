import numpy as np

rng = np.random.default_rng(42)


class SpeculativeDecoder:
    """
    Toy simulation of the modified rejection sampling scheme that lets a small
    draft model propose tokens for a large target model to verify in parallel,
    while guaranteeing the output distribution is EXACTLY the target's.
    """

    def __init__(self, target_dist, draft_dist):
        self.p = np.asarray(target_dist, dtype=float)
        self.q = np.asarray(draft_dist, dtype=float)
        assert np.isclose(self.p.sum(), 1.0) and np.isclose(self.q.sum(), 1.0)

    def acceptance_rate(self):
        """alpha = sum_x min(p(x), q(x)) = 1 - total_variation_distance(p, q)."""
        return np.sum(np.minimum(self.p, self.q))

    def residual_distribution(self):
        """p_res(x) = max(0, p(x) - q(x)) / sum_y max(0, p(y) - q(y))."""
        residual = np.maximum(0.0, self.p - self.q)
        return residual / residual.sum()

    def sample_one_token(self):
        """
        Single draft-then-verify step. Returns (token, was_accepted).
        This is the mechanism proven exact in the TIL note, Section 3.
        """
        draft_token = rng.choice(len(self.q), p=self.q)
        accept_prob = min(1.0, self.p[draft_token] / self.q[draft_token])

        if rng.random() < accept_prob:
            return draft_token, True

        residual = self.residual_distribution()
        resampled_token = rng.choice(len(self.p), p=residual)
        return resampled_token, False

    def simulate_speculative_block(self, k_draft_tokens):
        """
        Simulates one full speculative iteration: draft K tokens, verify them
        left-to-right, stop at first rejection (its resample IS the token
        emitted at that slot). If all K survive, the same target forward
        pass already priced in one extra 'bonus' token beyond the block at
        no additional HBM cost, so it's emitted for free. Returns tokens
        emitted this iteration, matching sum_{i=0}^{K} alpha^i in expectation.
        """
        emitted = 0
        all_accepted = True
        for _ in range(k_draft_tokens):
            _, accepted = self.sample_one_token()
            emitted += 1
            if not accepted:
                all_accepted = False
                break
        if all_accepted:
            emitted += 1  # free bonus token, already covered by the same forward pass
        return emitted


def verify_exactness(decoder, n_trials=200_000):
    """Monte Carlo check that sample_one_token()'s marginal == target p exactly."""
    vocab_size = len(decoder.p)
    counts = np.zeros(vocab_size)
    for _ in range(n_trials):
        token, _ = decoder.sample_one_token()
        counts[token] += 1
    empirical = counts / n_trials
    max_abs_error = np.max(np.abs(empirical - decoder.p))
    return empirical, max_abs_error


def verify_expected_tokens_per_iteration(decoder, k_draft_tokens, n_trials=50_000):
    """Monte Carlo check against the closed form (1 - alpha^(K+1)) / (1 - alpha)."""
    totals = [decoder.simulate_speculative_block(k_draft_tokens) for _ in range(n_trials)]
    empirical_mean = np.mean(totals)

    alpha = decoder.acceptance_rate()
    theoretical_mean = sum(alpha ** i for i in range(k_draft_tokens + 1))
    return empirical_mean, theoretical_mean, alpha


def expected_speedup(alpha, k_draft_tokens, cost_ratio):
    """Speedup = (1 - alpha^(K+1)) / ((1 - alpha) * (1 + K * cost_ratio))."""
    tokens_per_iter = (1 - alpha ** (k_draft_tokens + 1)) / (1 - alpha)
    iter_cost = 1 + k_draft_tokens * cost_ratio  # in units of T_target
    return tokens_per_iter / iter_cost


if __name__ == "__main__":
    # A misaligned-but-overlapping draft/target pair over a toy 6-token vocab.
    target_p = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
    draft_q = [0.20, 0.20, 0.20, 0.20, 0.15, 0.05]

    decoder = SpeculativeDecoder(target_p, draft_q)

    print("=" * 60)
    print("EXACTNESS CHECK — does sampling via draft+verify match target p?")
    print("=" * 60)
    empirical, max_err = verify_exactness(decoder)
    print(f"Target distribution p:     {np.round(decoder.p, 4)}")
    print(f"Empirical distribution:    {np.round(empirical, 4)}")
    print(f"Max absolute error:        {max_err:.5f}  (should shrink toward 0 with more trials)")

    print("\n" + "=" * 60)
    print("ACCEPTANCE RATE — alpha = 1 - total_variation_distance(p, q)")
    print("=" * 60)
    alpha = decoder.acceptance_rate()
    tv_dist = 0.5 * np.sum(np.abs(decoder.p - decoder.q))
    print(f"alpha (sum of min(p,q)):   {alpha:.4f}")
    print(f"1 - TV(p, q):              {1 - tv_dist:.4f}  (should match alpha exactly)")

    print("\n" + "=" * 60)
    print("EXPECTED TOKENS PER SPECULATIVE ITERATION vs. closed form")
    print("=" * 60)
    for k in [1, 2, 4, 8]:
        empirical_mean, theoretical_mean, a = verify_expected_tokens_per_iteration(decoder, k)
        print(f"K={k:>2}  empirical={empirical_mean:.3f}  theoretical=(1-a^(K+1))/(1-a)={theoretical_mean:.3f}")

    print("\n" + "=" * 60)
    print("SPEEDUP vs. K, at a few draft/target cost ratios (c = T_draft / T_target)")
    print("=" * 60)
    for cost_ratio in [0.05, 0.1, 0.2]:
        speedups = [float(expected_speedup(alpha, k, cost_ratio)) for k in range(1, 9)]
        best_k = int(np.argmax(speedups)) + 1
        print(f"c={cost_ratio:<5} best K={best_k}  "
              f"speedups(K=1..8)={[round(s, 2) for s in speedups]}")