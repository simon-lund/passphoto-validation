"""The German ID photo rules judged by OpenAI's Decisions API: the baseline passphoto-validation is measured against."""
from .report import Report, RuleResult, Thresholds, interpret, results_row
from .request import build_request, load_photo, questions, send
from .rules import GROUPS, RULES

__all__ = ['GROUPS', 'RULES', 'Report', 'RuleResult', 'Thresholds', 'build_request', 'interpret', 'load_photo',
           'questions', 'results_row', 'send']
