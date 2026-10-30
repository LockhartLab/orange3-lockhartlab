import unittest

import numpy as np

from orangecontrib.seqprops import properties as sp


def dna(seq, group):
    names, X, _ = sp.compute([sp.clean(seq)], sp.NUCLEOTIDE, [group])
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
        names, X, _ = sp.compute(["MKV"], sp.PROTEIN, [sp.PROPERTIES, sp.SKEW])
        self.assertEqual(names, sp.FEATURES[sp.PROTEIN][sp.PROPERTIES])
        self.assertEqual(X.shape, (1, 11))


if __name__ == "__main__":
    unittest.main()
