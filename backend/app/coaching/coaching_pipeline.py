"""Phase 4.5: combine descriptive form, sourced feedback and temporal evidence.

Does not determine whether a player matches a coach's form.
No model inference, no changes to the shot tracker, and no unverified drills.
"""
from app.coaching.form_comparison import FormComparisonAnalyzer
from app.coaching.coaching_feedback import CoachingFeedbackEngine
from app.coaching.temporal_form_analyzer import TemporalFormAnalyzer


class CoachingReportPipeline:
    def __init__(self):
        self.comparison = FormComparisonAnalyzer()
        self.feedback = CoachingFeedbackEngine()
        self.temporal = TemporalFormAnalyzer()

    def analyze(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected a full Phase 3 analysis payload")

        metrics = payload.get("metrics")
        pose_analysis = payload.get("analysis")
        if not isinstance(metrics, dict) or not isinstance(metrics.get("shots"), list):
            raise ValueError("Expected metrics.shots list")
        if not isinstance(pose_analysis, dict) or not isinstance(
            pose_analysis.get("frames"), list
        ):
            raise ValueError("Expected analysis.frames list")

        comparison = self.comparison.analyze(payload)
        coaching = self.feedback.analyze(comparison)
        temporal = self.temporal.analyze(payload)

        counts = (comparison["shot_count"], coaching["shot_count"],
                  temporal["shot_count"], len(metrics["shots"]))
        if len(set(counts)) != 1:
            raise ValueError("Coaching report shot counts do not match")

        quality_by_shot = []
        for shot in temporal["shots"]:
            segments = shot["segments"]
            phases = {
                key: {
                    "status": seg["quality"],
                    **seg["quality_metrics"],
                }
                for key, seg in segments.items()
            }
            quality_by_shot.append({
                "shot_number": shot["shot_number"],
                "arm": shot["arm"],
                "start_frame": shot["start_frame"],
                "release_frame": shot["release_frame"],
                "end_frame": shot["end_frame"],
                "timing_evidence": phases,
                "technique_judgment": "not_assessed",
            })

        return {
            "schema_version": "1.0",
            "shot_count": counts[0],
            "intended_form_match": "not_assessed",
            "official_endorsement": False,
            "first_review_suggestion": coaching["first_review_suggestion"],
            "shots": quality_by_shot,
            "comparison": comparison,
            "coaching_feedback": coaching,
            "temporal_analysis": temporal,
            "limitations": [
                "Frame coverage alone does not validate pose landmark accuracy.",
                "Technical correctness against SeeMikeDunn is not assessed.",
                "Engineering timing gates are not basketball coaching standards.",
                "Shot end_frame is not necessarily biomechanical follow-through.",
                "Missing video observations are not interpolated for form judgments.",
            ],
        }
