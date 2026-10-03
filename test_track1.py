import unittest

from track1_page_replacement import make_trace, simulate, train_learned_selector


class PageReplacementTests(unittest.TestCase):
    def test_trace_contains_two_equal_phases(self):
        trace, phases = make_trace()
        self.assertEqual(len(trace), 800)
        self.assertEqual(phases[:400], ["locality"] * 400)
        self.assertEqual(phases[400:], ["random_bursty"] * 400)

    def test_optimal_is_not_worse_than_lru_on_reference_trace(self):
        trace, phases = make_trace()
        optimal = simulate(trace, phases, "Optimal")
        lru = simulate(trace, phases, "LRU")
        self.assertLessEqual(sum(row.faults for row in optimal), sum(row.faults for row in lru))

    def test_learned_policy_runs_for_both_phases(self):
        training_trace, _ = make_trace(seed=308)
        trace, phases = make_trace()
        selector = train_learned_selector(training_trace)
        result = simulate(trace, phases, "Learned", learned_selector=selector)
        self.assertEqual({row.phase for row in result}, {"locality", "random_bursty"})
        self.assertEqual(sum(row.references for row in result), len(trace))


if __name__ == "__main__":
    unittest.main()
