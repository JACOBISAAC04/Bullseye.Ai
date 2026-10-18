"""Phase 5: compose validated shooting evidence into measured coaching reviews.

This layer is deliberately separate from YOLO/pose/shot detection. It never
edits source reports, reconstructs missing ball frames, or grades technique.
Run it with an API-shaped analysis dict and a ball-observation report.
"""

from __future__ import annotations

from app.coaching.temporal_form_analyzer import TemporalFormAnalyzer
from app.coaching.ball_evidence_adapter import BallEvidenceAdapter
from app.coaching.technique_signal_integration import TechniqueSignalIntegrator
from app.coaching.coaching_reference_registry import CoachingReferenceRegistry
from app.coaching.coaching_review_generator import CoachingReviewGenerator
from app.coaching.measurement_coaching_feedback import MeasurementCoachingFeedback


class MeasurementCoachingPipeline:
    """Generate auditable intermediate reports and conservative final feedback."""

    def analyze(self, api_result: dict, ball_observations: dict) -> dict:
        if not isinstance(api_result, dict) or not isinstance(ball_observations, dict):
            raise ValueError("Analysis and ball observations must be JSON objects")

        temporal = TemporalFormAnalyzer().analyze(api_result)
        mechanics = BallEvidenceAdapter().analyze(api_result, ball_observations)
        integrated = TechniqueSignalIntegrator().analyze(temporal, mechanics)

        registry = CoachingReferenceRegistry()
        sourced = registry.build_report(integrated)
        reviews = CoachingReviewGenerator(registry=registry).generate(sourced)
        measured = MeasurementCoachingFeedback().generate(
            temporal, mechanics, reviews
        )

        expected = api_result.get("shot_count")
        if type(expected) is not int:
            raise ValueError("Input shot_count must be an integer")
        for name, report in (
            ("temporal", temporal),
            ("mechanics", mechanics),
            ("integrated", integrated),
            ("registry", sourced),
            ("review", reviews),
            ("measured", measured),
        ):
            if report.get("shot_count") != expected or len(report.get("shots", [])) != expected:
                raise ValueError(f"{name}: shot count changed")
            if report.get("intended_form_match") != "not_assessed":
                raise ValueError(f"{name}: unapproved form assessment")
            if report.get("official_endorsement") is not False:
                raise ValueError(f"{name}: official endorsement must be false")
        if measured.get("corrections_generated") != 0:
            raise ValueError("Automatic technical corrections are not permitted")

        return {
            "temporal": temporal,
            "mechanics": mechanics,
            "technique": integrated,
            "sources": sourced,
            "review": reviews,
            "measured": measured,
        }
