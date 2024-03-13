from arcgis.learn._utils.env import _LAMBDA_TEXT_CLASSIFICATION
from ._text_classifier import TextClassifier

if not _LAMBDA_TEXT_CLASSIFICATION:
    from ._ner import EntityRecognizer
    from ._seq2seq import SequenceToSequence
    from ._qna import QuestionAnswering
    from ._fill_mask import FillMask
    from ._summarization import TextSummarizer
    from ._text_generation import TextGenerator
    from ._translation import TextTranslator
    from ._zero_shot_classifier import ZeroShotClassifier
