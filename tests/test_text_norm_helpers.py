from __future__ import annotations

import unittest

from src.data.text_norm import get_text_norm
from src.data.text_norm.remove_disfluencies import (
    is_valid_word,
    norm_string,
    remove_disfluencies,
)


class TextNormalizationTests(unittest.TestCase):
    def test_remove_disfluencies_drops_fillers(self):
        self.assertEqual(remove_disfluencies("um hello uh there"), "hello there")

    def test_is_valid_word_identifies_supported_patterns(self):
        self.assertEqual(is_valid_word("3.5%"), (True, "number_and_percentage"))
        self.assertEqual(is_valid_word("test-site"), (True, "word_with_hyphen"))
        self.assertEqual(is_valid_word("p.m"), (True, "abbreviation"))

    def test_norm_string_applies_expected_transformations(self):
        self.assertEqual(norm_string("test-site 3.5%"), "TEST SITE 3 POINT 5 PERCENT")

    def test_get_text_norm_builds_disfluency_removing_whisper_normalizer(self):
        normalizer = get_text_norm("whisper_basic_rm_disf")
        self.assertEqual(normalizer("um hello"), "hello")

    def test_get_text_norm_rejects_unknown_type(self):
        with self.assertRaisesRegex(ValueError, "Unsupported text normalization type"):
            get_text_norm("not-a-real-normalizer")


if __name__ == "__main__":
    unittest.main()
