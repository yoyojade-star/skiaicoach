"""Unit tests for `schema` Pydantic models."""
from __future__ import annotations

import json
import unittest

import pytest

pytest.importorskip("pydantic")
from pydantic import ValidationError

from schema import SkiCoachingFeedback


def _valid_payload() -> dict:
    return {
        "primary_fault": "Backseat",
        "biomechanical_explanation": "Hips behind boots reduces fore-aft balance.",
        "carving_score_analysis": "Score 45/100; increase edge angle earlier in turn.",
        "recommended_drill_name": "Thousand Steps",
        "drill_steps": ["Step 1", "Step 2"],
        "visual_observations": "Hands low in transition.",
        "progression_note": "First run on file.",
    }


class TestSkiCoachingFeedbackValidation(unittest.TestCase):
    def test_model_validate_accepts_complete_dict(self):
        data = _valid_payload()
        m = SkiCoachingFeedback.model_validate(data)
        self.assertEqual(m.primary_fault, "Backseat")
        self.assertEqual(m.drill_steps, ["Step 1", "Step 2"])

    def test_model_validate_json_round_trip(self):
        data = _valid_payload()
        raw = json.dumps(data)
        m = SkiCoachingFeedback.model_validate_json(raw)
        self.assertEqual(m.model_dump(), data)

    def test_model_dump_matches_input_keys(self):
        data = _valid_payload()
        m = SkiCoachingFeedback.model_validate(data)
        dumped = m.model_dump()
        self.assertEqual(set(dumped.keys()), set(data.keys()))

    def test_missing_required_field_raises(self):
        data = _valid_payload()
        del data["primary_fault"]
        with self.assertRaises(ValidationError) as ctx:
            SkiCoachingFeedback.model_validate(data)
        err = ctx.exception
        self.assertTrue(any("primary_fault" in str(e) for e in err.errors()))

    def test_drill_steps_must_be_list_of_strings(self):
        data = _valid_payload()
        data["drill_steps"] = "not a list"
        with self.assertRaises(ValidationError):
            SkiCoachingFeedback.model_validate(data)

    def test_empty_drill_steps_allowed(self):
        data = _valid_payload()
        data["drill_steps"] = []
        m = SkiCoachingFeedback.model_validate(data)
        self.assertEqual(m.drill_steps, [])


class TestSkiCoachingFeedbackJsonSchema(unittest.TestCase):
    def test_model_json_schema_has_expected_properties(self):
        schema = SkiCoachingFeedback.model_json_schema()
        self.assertEqual(schema.get("title"), "SkiCoachingFeedback")
        props = schema.get("properties", {})
        for key in (
            "primary_fault",
            "biomechanical_explanation",
            "carving_score_analysis",
            "recommended_drill_name",
            "drill_steps",
            "visual_observations",
            "progression_note",
        ):
            self.assertIn(key, props, msg=f"missing property {key!r}")

    def test_drill_steps_schema_is_array(self):
        schema = SkiCoachingFeedback.model_json_schema()
        steps = schema["properties"]["drill_steps"]
        self.assertEqual(steps.get("type"), "array")


if __name__ == "__main__":
    unittest.main()
