import unittest

import numpy as np

from orangecontrib.seqprops import properties as sp


def dna(seq, group):
    names, X, _, _ = sp.compute([sp.clean(seq)], sp.NUCLEOTIDE, [group])
    return dict(zip(names, X[0]))


class TestNucleotideFeatures(unittest.TestCase):
    def test_skew(self):
        f = dna("GGGCAAAT", sp.SKEW)
        self.assertAlmostEqual(f["GC skew"], (3 - 1) / 4)
        self.assertAlmostEqual(f["AT skew"], (3 - 1) / 4)

    def test_cpg(self):
        f = dna("CGCGNNNN", sp.SKEW)
        # N is ignored: 2 CG among 4 valid bases with C=2, G=2: 2 * 4 / (2 * 2)
        self.assertAlmostEqual(f["CpG obs/exp"], 2.0)
        self.assertTrue(np.isnan(dna("AAAA", sp.SKEW)["CpG obs/exp"]))
        self.assertTrue(np.isnan(dna("AAAA", sp.SKEW)["GC skew"]))

    def test_rna_treated_like_dna(self):
        self.assertEqual(dna("GGGCAAAU", sp.SKEW), dna("GGGCAAAT", sp.SKEW))

    def test_counts(self):
        self.assertEqual(len(sp.feature_names(sp.PROTEIN)), 11)
        self.assertEqual(len(sp.feature_names(sp.NUCLEOTIDE)), 7)

    def test_protein_ignores_dna_groups(self):
        names, X, _, _ = sp.compute(["MKV"], sp.PROTEIN, [sp.PROPERTIES, sp.SKEW])
        self.assertEqual(names, sp.FEATURES[sp.PROTEIN][sp.PROPERTIES])
        self.assertEqual(X.shape, (1, 11))

    def test_bad_sequences_are_reported_not_raised(self):
        # the bug report: a protein read as DNA keeps only A/C/G/T/U and can
        # end up mixing T and U, which Biopython rejected with a crash
        names, X, empty, failed = sp.compute(["ACGT", "MKTUAGCRV", "", "GGCC"],
                                             sp.NUCLEOTIDE)
        self.assertEqual(empty, [2])
        self.assertEqual([i for i, _ in failed], [1])
        self.assertIn("both T and U", failed[0][1])
        self.assertTrue(np.isnan(X[1]).all() and np.isnan(X[2]).all())
        self.assertEqual(X[[0, 3], 0].tolist(), [4, 4])

    def test_non_nucleotide_fraction(self):
        self.assertEqual(sp.non_nucleotide_fraction(["ACGT", "ACGUN"]), 0)
        self.assertGreater(sp.non_nucleotide_fraction(["MKTAYIAKQRQISFVKSHFSRQ"]), 0.5)
        self.assertEqual(sp.non_nucleotide_fraction(["", ""]), 0)


if __name__ == "__main__":
    unittest.main()
