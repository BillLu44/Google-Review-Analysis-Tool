import os, sys
# Insert project root (one level up) into module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, MagicMock, call

from pipeline.absa import (
    extract_aspects,
    analyze_aspect_sentiment,
    analyze_aspect_sentiment_with_clauses,
    analyze_absa,
    extract_aspect_context,
    analyze_complex_aspects
)

# Path for patching module-level variables
ABSA_MODULE_PATH = 'pipeline.absa'

class TestExtractAspects(unittest.TestCase):
    def test_extract_simple(self):
        self.assertEqual(sorted(extract_aspects("The food was great.")), sorted(["food"]))

    def test_extract_multiple_aspects(self):
        self.assertEqual(sorted(extract_aspects("Great food and excellent service.")), sorted(["food", "service"]))

    def test_extract_plural_and_singular(self):
        # "dishes" should be preferred over "dish"
        self.assertEqual(sorted(extract_aspects("The dish was good, the dishes were amazing.")), sorted(["dishes"]))
        self.assertEqual(sorted(extract_aspects("The server was nice, the servers were quick.")), sorted(["servers"]))

    def test_extract_no_common_aspects(self):
        self.assertEqual(extract_aspects("A lovely evening."), ["overall"])

    def test_extract_empty_text(self):
        self.assertEqual(extract_aspects(""), ["overall"])
        self.assertEqual(extract_aspects("   "), ["overall"])

    def test_case_insensitivity(self):
        self.assertEqual(sorted(extract_aspects("The FOOD was great.")), sorted(["food"]))

    def test_aspect_substring_of_another(self):
        # Ensure "server" isn't extracted if "servers" is present and "server" is part of "servers"
        # The current logic handles this by set operations and preferring plurals.
        self.assertEqual(sorted(extract_aspects("The servers and server were good.")), sorted(["servers"]))
        # Test with non-plural relation, e.g. "price" and "pricing"
        self.assertEqual(sorted(extract_aspects("The price and pricing were fair.")), sorted(["price", "pricing"])) # Expect both if both are in COMMON_ASPECTS

    def test_aspect_with_punctuation(self):
        self.assertEqual(sorted(extract_aspects("Food! It was good.")), sorted(["food"]))
        self.assertEqual(sorted(extract_aspects("The service? Impeccable.")), sorted(["service"]))


class TestAnalyzeAspectSentiment(unittest.TestCase):
    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_positive_sentiment_label_format(self, mock_classifier, mock_logger):
        mock_classifier.return_value = [[{'label': 'LABEL_2', 'score': 0.95}]]
        result = analyze_aspect_sentiment("The food is great.", "food")
        self.assertEqual(result['sentiment_label'], 'positive')
        self.assertEqual(result['sentiment_score'], 0.95)
        self.assertEqual(result['confidence'], 0.95)
        mock_classifier.assert_called_once_with("The food is great. [SEP] food")

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_negative_sentiment_direct_label(self, mock_classifier, mock_logger):
        mock_classifier.return_value = [[{'label': 'negative', 'score': 0.88}]]
        result = analyze_aspect_sentiment("The service is bad.", "service")
        self.assertEqual(result['sentiment_label'], 'negative')
        self.assertEqual(result['sentiment_score'], -0.88)
        self.assertEqual(result['confidence'], 0.88)
        mock_classifier.assert_called_once_with("The service is bad. [SEP] service")

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_neutral_sentiment(self, mock_classifier, mock_logger):
        mock_classifier.return_value = [[{'label': 'LABEL_1', 'score': 0.7}]]
        result = analyze_aspect_sentiment("The price is average.", "price")
        self.assertEqual(result['sentiment_label'], 'neutral')
        self.assertEqual(result['sentiment_score'], 0.0)
        self.assertEqual(result['confidence'], 0.7)

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier', None) # Simulate classifier not loaded
    def test_classifier_is_none(self, mock_logger):
        result = analyze_aspect_sentiment("Any text", "aspect")
        self.assertEqual(result['sentiment_label'], 'neutral')
        self.assertEqual(result['sentiment_score'], 0.0)
        mock_logger.warning.assert_called_with("ABSA classifier not available. Cannot analyze aspect 'aspect'.")

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_classifier_returns_empty_list(self, mock_classifier, mock_logger):
        mock_classifier.return_value = [] # Empty list
        result = analyze_aspect_sentiment("Any text", "aspect")
        self.assertEqual(result['sentiment_label'], 'neutral')
        self.assertEqual(result['sentiment_score'], 0.0)
        mock_logger.error.assert_called_with("Error analyzing aspect sentiment for 'aspect': empty ABSA output")

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_classifier_returns_empty_nested_list(self, mock_classifier, mock_logger):
        mock_classifier.return_value = [[]] # Empty nested list
        result = analyze_aspect_sentiment("Any text", "aspect")
        self.assertEqual(result['sentiment_label'], 'neutral')
        self.assertEqual(result['sentiment_score'], 0.0)
        mock_logger.error.assert_called_with("Error analyzing aspect sentiment for 'aspect': empty ABSA output")

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_classifier_raises_exception(self, mock_classifier, mock_logger):
        mock_classifier.side_effect = Exception("Model failure")
        result = analyze_aspect_sentiment("Any text", "aspect")
        self.assertEqual(result['sentiment_label'], 'neutral')
        self.assertEqual(result['sentiment_score'], 0.0)
        mock_logger.error.assert_called_with("Error analyzing aspect sentiment for 'aspect': Model failure")

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_ambiguous_label_defaults_to_neutral(self, mock_classifier, mock_logger):
        mock_classifier.return_value = [[{'label': 'unknown_sentiment', 'score': 0.6}]]
        result = analyze_aspect_sentiment("The decor was something.", "decor")
        self.assertEqual(result['sentiment_label'], 'neutral')
        self.assertEqual(result['sentiment_score'], 0.0) # Score becomes 0.0 for neutral
        self.assertEqual(result['confidence'], 0.6)

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_direct_list_of_dicts_result(self, mock_classifier, mock_logger):
        mock_classifier.return_value = [{'label': 'LABEL_2', 'score': 0.9}] # Not nested
        result = analyze_aspect_sentiment("The food is great.", "food")
        self.assertEqual(result['sentiment_label'], 'positive')
        self.assertEqual(result['sentiment_score'], 0.9)

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_result_dict_missing_score(self, mock_classifier, mock_logger):
        mock_classifier.return_value = [[{'label': 'LABEL_2'}]] # Missing score
        result = analyze_aspect_sentiment("Text", "aspect")
        self.assertEqual(result['sentiment_label'], 'neutral') # Defaults due to error
        self.assertEqual(result['sentiment_score'], 0.0)
        mock_logger.error.assert_called()

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_result_dict_missing_label(self, mock_classifier, mock_logger):
        mock_classifier.return_value = [[{'score': 0.9}]] # Missing label
        result = analyze_aspect_sentiment("Text", "aspect")
        self.assertEqual(result['sentiment_label'], 'neutral') # Defaults due to error
        self.assertEqual(result['sentiment_score'], 0.0)
        mock_logger.error.assert_called()


class TestAnalyzeAspectSentimentWithClauses(unittest.TestCase):
    # Assuming analyze_aspect_sentiment_with_clauses is completed to process
    # classifier results for each clause similar to analyze_aspect_sentiment.

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_contradictory_clauses_food(self, mock_classifier, mock_logger):
        text = "The food was amazing but the food was also too salty."
        aspect = "food"
        
        def mock_side_effect(clause_input):
            if "amazing" in clause_input: # "The food was amazing [SEP] food"
                return [[{'label': 'LABEL_2', 'score': 0.9}]] 
            elif "salty" in clause_input: # "the food was also too salty [SEP] food"
                return [[{'label': 'LABEL_0', 'score': 0.8}]]
            return []

        mock_classifier.side_effect = mock_side_effect
        results = analyze_aspect_sentiment_with_clauses(text, aspect)
        
        self.assertEqual(len(results), 2)
        # Clause 1: "The food was amazing"
        self.assertEqual(results[0]['aspect'], 'food')
        self.assertEqual(results[0]['sentiment_label'], 'positive')
        self.assertEqual(results[0]['sentiment_score'], 0.9)
        self.assertEqual(results[0]['clause'], "The food was amazing")
        # Clause 2: "the food was also too salty"
        self.assertEqual(results[1]['aspect'], 'food')
        self.assertEqual(results[1]['sentiment_label'], 'negative')
        self.assertEqual(results[1]['sentiment_score'], -0.8)
        self.assertEqual(results[1]['clause'], "the food was also too salty")
        
        expected_calls = [
            call("The food was amazing [SEP] food"),
            call("the food was also too salty [SEP] food")
        ]
        mock_classifier.assert_has_calls(expected_calls, any_order=False)


    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_single_clause_no_but(self, mock_classifier, mock_logger):
        text = "The service was excellent."
        aspect = "service"
        mock_classifier.return_value = [[{'label': 'LABEL_2', 'score': 0.95}]]
        results = analyze_aspect_sentiment_with_clauses(text, aspect)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['sentiment_label'], 'positive')
        self.assertEqual(results[0]['clause'], text)
        mock_classifier.assert_called_once_with(f"{text} [SEP] {aspect}")

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier', None)
    def test_classifier_none_with_clauses(self, mock_logger):
        text = "Some text"
        aspect = "aspect"
        results = analyze_aspect_sentiment_with_clauses(text, aspect)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['sentiment_label'], 'neutral')
        self.assertEqual(results[0]['aspect'], aspect)
        # Assuming the completed function adds the original text as 'clause' in this default case
        self.assertEqual(results[0]['clause'], text) 

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_aspect_in_one_clause_only(self, mock_classifier, mock_logger):
        text = "The food was great, but the music was too loud."
        aspect = "food"
        # Mock classifier to return positive for the clause containing "food"
        mock_classifier.return_value = [[{'label': 'LABEL_2', 'score': 0.85}]]
        
        results = analyze_aspect_sentiment_with_clauses(text, aspect)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['aspect'], 'food')
        self.assertEqual(results[0]['sentiment_label'], 'positive')
        self.assertEqual(results[0]['sentiment_score'], 0.85)
        self.assertEqual(results[0]['clause'], "The food was great") # Assumes the first part is the relevant clause
        mock_classifier.assert_called_once_with("The food was great [SEP] food")

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_aspect_not_in_any_clause_after_split(self, mock_classifier, mock_logger):
        text = "The service was slow, but the decor was nice."
        aspect = "food" # Aspect not present
        
        results = analyze_aspect_sentiment_with_clauses(text, aspect)
        
        # Expect default neutral result as aspect is not found in any clause
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['aspect'], 'food')
        self.assertEqual(results[0]['sentiment_label'], 'neutral')
        self.assertEqual(results[0]['sentiment_score'], 0.0)
        # Depending on implementation, 'clause' might be the original text or None/empty.
        # The current absa.py returns a default without a clause if no mentions.
        # Let's assume it should be the original text if no specific clause was analyzed.
        # self.assertEqual(results[0].get('clause'), text) # Or check if 'clause' key is absent or None
        self.assertNotIn('clause', results[0] ) # Or specific default like results[0]['clause'] == text
        mock_classifier.assert_not_called()

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier')
    def test_clause_analysis_error_handling(self, mock_classifier, mock_logger):
        text = "The food was good but then something broke."
        aspect = "food"
        mock_classifier.side_effect = Exception("Classifier error during clause analysis")
        
        results = analyze_aspect_sentiment_with_clauses(text, aspect)
        
        self.assertEqual(len(results), 1) # Should return a default neutral result for the aspect
        self.assertEqual(results[0]['aspect'], aspect)
        self.assertEqual(results[0]['sentiment_label'], 'neutral')
        self.assertEqual(results[0]['sentiment_score'], 0.0)
        mock_logger.error.assert_called()


class TestAnalyzeAbsa(unittest.TestCase):
    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier', new_callable=MagicMock) # Ensure classifier is mocked
    @patch(f'{ABSA_MODULE_PATH}.extract_aspects')
    @patch(f'{ABSA_MODULE_PATH}.analyze_aspect_sentiment_with_clauses')
    def test_no_aspects_found_defaults_to_overall(self, mock_analyze_clauses, mock_extract_aspects, mock_absa_classifier, mock_logger):
        mock_extract_aspects.return_value = ["overall"]
        mock_analyze_clauses.return_value = [{'aspect': 'overall', 'sentiment_label': 'positive', 'sentiment_score': 0.8, 'confidence': 0.8, 'clause': 'It was a good day.'}]
        
        result = analyze_absa("It was a good day.")
        
        self.assertEqual(len(result['aspect_details']), 1)
        self.assertEqual(result['aspect_details'][0]['aspect'], 'overall')
        self.assertEqual(result['aspect_details'][0]['sentiment_label'], 'positive')
        self.assertEqual(result['num_pos_aspects'], 1) # This depends on your aggregation logic in analyze_absa
        self.assertEqual(result['num_neg_aspects'], 0)
        self.assertAlmostEqual(result['avg_aspect_score'], 0.8)
        mock_extract_aspects.assert_called_once_with("It was a good day.")
        mock_analyze_clauses.assert_called_once_with("It was a good day.", "overall")

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier', new_callable=MagicMock)
    @patch(f'{ABSA_MODULE_PATH}.extract_aspects')
    @patch(f'{ABSA_MODULE_PATH}.analyze_aspect_sentiment_with_clauses')
    def test_single_aspect_single_mention(self, mock_analyze_clauses, mock_extract_aspects, mock_absa_classifier, mock_logger):
        mock_extract_aspects.return_value = ["food"]
        mock_analyze_clauses.return_value = [{'aspect': 'food', 'sentiment_label': 'negative', 'sentiment_score': -0.7, 'confidence': 0.7, 'clause': 'The food was bad.'}]
        
        result = analyze_absa("The food was bad.")
        
        self.assertEqual(len(result['aspect_details']), 1)
        self.assertEqual(result['aspect_details'][0]['aspect'], 'food')
        self.assertEqual(result['aspect_details'][0]['sentiment_label'], 'negative')
        self.assertEqual(result['num_neg_aspects'], 1)
        self.assertEqual(result['num_pos_aspects'], 0)
        self.assertAlmostEqual(result['avg_aspect_score'], -0.7)

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier', new_callable=MagicMock)
    @patch(f'{ABSA_MODULE_PATH}.extract_aspects')
    @patch(f'{ABSA_MODULE_PATH}.analyze_aspect_sentiment_with_clauses')
    def test_multiple_aspects(self, mock_analyze_clauses, mock_extract_aspects, mock_absa_classifier, mock_logger):
        mock_extract_aspects.return_value = ["food", "service"]
        
        def analyze_clauses_side_effect(text, aspect):
            if aspect == "food":
                return [{'aspect': 'food', 'sentiment_label': 'positive', 'sentiment_score': 0.9, 'confidence': 0.9, 'clause': 'Food great'}]
            if aspect == "service":
                return [{'aspect': 'service', 'sentiment_label': 'negative', 'sentiment_score': -0.6, 'confidence': 0.6, 'clause': 'Service bad'}]
            return []
        mock_analyze_clauses.side_effect = analyze_clauses_side_effect
        
        result = analyze_absa("Food great, service bad.")
        
        self.assertEqual(len(result['aspect_details']), 2)
        self.assertEqual(result['num_pos_aspects'], 1)
        self.assertEqual(result['num_neg_aspects'], 1)
        self.assertAlmostEqual(result['avg_aspect_score'], (0.9 - 0.6) / 2)
        self.assertAlmostEqual(result['avg_aspect_confidence'], (0.9 + 0.6) / 2)

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier', new_callable=MagicMock) # Mock the classifier itself
    @patch(f'{ABSA_MODULE_PATH}.extract_aspects')
    @patch(f'{ABSA_MODULE_PATH}.analyze_aspect_sentiment_with_clauses')
    def test_single_aspect_multiple_mentions_aggregation(self, mock_analyze_clauses, mock_extract_aspects, mock_absa_classifier_instance, mock_logger):
        """
        Test aggregation when one aspect has multiple mentions (e.g., from different clauses).
        This test assumes your analyze_absa function correctly implements the aggregation logic
        for num_pos_aspects, num_neg_aspects, and the final sentiment_label for the aspect.
        """
        mock_extract_aspects.return_value = ["food"]
        # analyze_aspect_sentiment_with_clauses returns a list of mentions
        mock_analyze_clauses.return_value = [
            {'aspect': 'food', 'sentiment_label': 'positive', 'sentiment_score': 0.9, 'confidence': 0.9, 'clause': 'The food was great'},
            {'aspect': 'food', 'sentiment_label': 'negative', 'sentiment_score': -0.5, 'confidence': 0.5, 'clause': 'but the food was also cold'}
        ]
        
        # Ensure the global absa_classifier is not None for this test if analyze_absa checks it
        # If analyze_absa uses the global directly, you might need to patch it there too or ensure it's set.
        # For simplicity, we assume analyze_absa relies on analyze_aspect_sentiment_with_clauses which uses the mock.

        result = analyze_absa("The food was great but the food was also cold.")
        
        # Assuming your aggregation logic in analyze_absa is:
        # - num_pos_aspects and num_neg_aspects count individual mentions.
        # - The 'aspect_details' entry for 'food' reflects an aggregated sentiment.
        #   (e.g., average score, and label based on that average)
        
        self.assertEqual(result['num_pos_aspects'], 1) # Counts the positive mention
        self.assertEqual(result['num_neg_aspects'], 1) # Counts the negative mention
        
        self.assertEqual(len(result['aspect_details']), 1)
        food_detail = result['aspect_details'][0]
        self.assertEqual(food_detail['aspect'], 'food')
        
        # Expected aggregated score: (0.9 - 0.5) / 2 = 0.2
        self.assertAlmostEqual(food_detail['sentiment_score'], 0.2)
        # Expected aggregated confidence: (0.9 + 0.5) / 2 = 0.7
        self.assertAlmostEqual(food_detail['confidence'], 0.7)
        # Expected label based on aggregated score (0.2 is positive)
        self.assertEqual(food_detail['sentiment_label'], 'positive') # This depends on thresholds in analyze_absa

        self.assertAlmostEqual(result['avg_aspect_score'], 0.2) # Since only one aspect overall
        self.assertAlmostEqual(result['avg_aspect_confidence'], 0.7)


    @patch(f'{ABSA_MODULE_PATH}.logger')
    def test_empty_input_text(self, mock_logger):
        result = analyze_absa("")
        expected = {
            'aspect_details': [], 'num_pos_aspects': 0, 'num_neg_aspects': 0,
            'avg_aspect_score': 0.0, 'avg_aspect_confidence': 0.0
        }
        self.assertEqual(result, expected)
        mock_logger.warning.assert_called_with("Empty or invalid text provided to ABSA analysis")

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.absa_classifier', None)
    def test_absa_classifier_none(self, mock_logger):
        # analyze_absa should check if absa_classifier is None at the beginning
        result = analyze_absa("Some text.")
        expected = {
            'aspect_details': [], 'num_pos_aspects': 0, 'num_neg_aspects': 0,
            'avg_aspect_score': 0.0, 'avg_aspect_confidence': 0.0
        }
        self.assertEqual(result, expected)
        mock_logger.warning.assert_called_with("ABSA classifier not available. Skipping ABSA.")


class TestExtractAspectContext(unittest.TestCase):
    def test_extract_context_basic(self):
        text = "The food was absolutely great and tasty for sure."
        aspect = "great"
        # window=3 means 3 words before "great" and 3 words after "great"
        # "food was absolutely" (before) + "great" + "and tasty for" (after)
        expected = "food was absolutely great and tasty for" 
        self.assertEqual(extract_aspect_context(text, aspect, window=3).strip(), expected)

    def test_extract_context_start(self):
        text = "Food was great and tasty."
        aspect = "Food"
        expected = "Food was great and" # window=3
        self.assertEqual(extract_aspect_context(text, aspect, window=3).strip(), expected)

    def test_extract_context_end(self):
        text = "Service was great."
        aspect = "great"
        expected = "Service was great" # window=3
        self.assertEqual(extract_aspect_context(text, aspect, window=3).strip(), expected)
    
    def test_extract_context_aspect_not_found(self):
        text = "Service was good."
        aspect = "food"
        expected = ""
        self.assertEqual(extract_aspect_context(text, aspect, window=3).strip(), expected)

    def test_extract_context_multiple_occurrences(self):
        # Current implementation joins contexts of all occurrences.
        text = "The food was good. Later the food was tasty."
        aspect = "food"
        # window=1: "The food was" and "the food was"
        expected = "The food was the food was" 
        self.assertEqual(extract_aspect_context(text, aspect, window=1).strip(), expected)

    def test_extract_context_window_larger_than_text(self):
        text = "Short food text"
        aspect = "food"
        expected = "Short food text"
        self.assertEqual(extract_aspect_context(text, aspect, window=10).strip(), expected)

    def test_extract_context_empty_text(self):
        text = ""
        aspect = "food"
        expected = ""
        self.assertEqual(extract_aspect_context(text, aspect).strip(), expected)

    def test_extract_context_empty_aspect(self):
        # If aspect is empty, `aspect.lower() in word.lower()` is true for non-empty words.
        # The current implementation would return the text repeated if not handled.
        # Let's assume it should return empty or original text.
        # For now, testing current behavior which might be "word1 word2 word1 word2 ..."
        # This test might indicate a need to refine extract_aspect_context for empty aspect.
        text = "Some text here"
        aspect = ""
        # Expected behavior for empty aspect might be to return the original text or empty.
        # Given current code: it will find "" in every word, so context for every word.
        # Let's assume the desired behavior for an empty aspect is to return an empty string.
        # To test current behavior:
        # words = text.split()
        # expected_parts = []
        # for _ in words:
        #     expected_parts.extend(words) # each word gets full text as context with window=len(words)
        # self.assertEqual(extract_aspect_context(text, aspect, window=len(words)).strip(), " ".join(expected_parts))
        # For a more robust test, let's assert it returns empty if aspect is empty.
        # This would require a change in extract_aspect_context:
        # if not aspect: return ""
        self.assertEqual(extract_aspect_context(text, aspect, window=3).strip(), "")


class TestAnalyzeComplexAspects(unittest.TestCase):
    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.extract_aspect_context')
    @patch(f'{ABSA_MODULE_PATH}.analyze_aspect_sentiment')
    def test_sentiment_flip_sarcastic_negative_context(self, mock_analyze_sentiment, mock_extract_context, mock_logger):
        mock_extract_context.return_value = "oh wonderful only bad food"
        mock_analyze_sentiment.return_value = {'aspect': 'food', 'sentiment_label': 'positive', 'sentiment_score': 0.8, 'confidence': 0.8}
        
        result = analyze_complex_aspects("Some text", "food")
        
        self.assertEqual(result['sentiment_label'], 'negative')
        self.assertEqual(result['sentiment_score'], -0.8) # Flipped score
        mock_analyze_sentiment.assert_called_once_with("Some text", "food")
        mock_extract_context.assert_called_once_with("Some text", "food")

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.extract_aspect_context')
    @patch(f'{ABSA_MODULE_PATH}.analyze_aspect_sentiment')
    def test_no_flip_sarcastic_only(self, mock_analyze_sentiment, mock_extract_context, mock_logger):
        mock_extract_context.return_value = "oh wonderful good food"
        mock_analyze_sentiment.return_value = {'aspect': 'food', 'sentiment_label': 'positive', 'sentiment_score': 0.8, 'confidence': 0.8}
        
        result = analyze_complex_aspects("Some text", "food")
        
        self.assertEqual(result['sentiment_label'], 'positive') # No flip
        self.assertEqual(result['sentiment_score'], 0.8)

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.extract_aspect_context')
    @patch(f'{ABSA_MODULE_PATH}.analyze_aspect_sentiment')
    def test_no_flip_negative_context_only(self, mock_analyze_sentiment, mock_extract_context, mock_logger):
        mock_extract_context.return_value = "only bad food"
        mock_analyze_sentiment.return_value = {'aspect': 'food', 'sentiment_label': 'positive', 'sentiment_score': 0.8, 'confidence': 0.8}
        
        result = analyze_complex_aspects("Some text", "food")
        
        self.assertEqual(result['sentiment_label'], 'positive') # No flip
        self.assertEqual(result['sentiment_score'], 0.8)

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.extract_aspect_context')
    @patch(f'{ABSA_MODULE_PATH}.analyze_aspect_sentiment')
    def test_no_flip_no_indicators(self, mock_analyze_sentiment, mock_extract_context, mock_logger):
        mock_extract_context.return_value = "good food"
        mock_analyze_sentiment.return_value = {'aspect': 'food', 'sentiment_label': 'positive', 'sentiment_score': 0.8, 'confidence': 0.8}
        
        result = analyze_complex_aspects("Some text", "food")
        
        self.assertEqual(result['sentiment_label'], 'positive') # No flip
        self.assertEqual(result['sentiment_score'], 0.8)

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.extract_aspect_context')
    @patch(f'{ABSA_MODULE_PATH}.analyze_aspect_sentiment')
    def test_complex_aspect_calls_base_sentiment(self, mock_analyze_sentiment, mock_extract_context, mock_logger):
        mock_analyze_sentiment.return_value = {'aspect': 'food', 'sentiment_label': 'neutral', 'sentiment_score': 0.0, 'confidence': 0.0}
        mock_extract_context.return_value = ""
        analyze_complex_aspects("Some text", "food")
        mock_analyze_sentiment.assert_called_once_with("Some text", "food")

    @patch(f'{ABSA_MODULE_PATH}.logger')
    @patch(f'{ABSA_MODULE_PATH}.extract_aspect_context')
    @patch(f'{ABSA_MODULE_PATH}.analyze_aspect_sentiment')
    def test_complex_aspect_calls_extract_context(self, mock_analyze_sentiment, mock_extract_context, mock_logger):
        mock_analyze_sentiment.return_value = {'aspect': 'food', 'sentiment_label': 'neutral', 'sentiment_score': 0.0, 'confidence': 0.0}
        mock_extract_context.return_value = ""
        analyze_complex_aspects("Some text", "food")
        mock_extract_context.assert_called_once_with("Some text", "food")


if __name__ == '__main__':
    # Define the output directory and file
    # Assuming your project root is one level up from the 'tests' directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist
    output_file_path = os.path.join(output_dir, 'test_absa_results.txt')

    # Open the output file
    with open(output_file_path, 'w') as f:
        # Create a TestSuite
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
        # Create a TextTestRunner with the file stream and verbosity
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        # Run the tests
        result = runner.run(suite)
    
    # Optionally, print a message to the console indicating where results are saved
    print(f"Test results saved to: {output_file_path}")

    # Exit with an appropriate code (0 for success, 1 for failure)
    # This is useful if you run this script as part of an automated process
    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)