"""Validation for private case configuration.

The paper algorithms obtain case-specific values through :func:`config.load_case`.
Clinical values are deliberately absent from this tracked module: authorised local
runs define them in the Git-ignored ``config/local.py``. The public synthetic case
is defined independently in ``examples/phantom/config_phantom.py``.
"""

from collections.abc import Mapping, Sequence
from numbers import Real


REQUIRED_CASE_FIELDS = (
    "dataset_id",
    "sim_folder",
    "venc_series",
    "venc_instance",
    "heart_rate_bpm",
    "systolic_mmHg",
    "diastolic_mmHg",
    "outlet_mean_flow",
    "outlet_branch_order",
    "rings_ignored",
    "mri_translation",
    "mri_rotation_deg",
    "mri_outlet_sections",
    "wall_cycle_closure",
    "stiffness_target_phase_index",
    "prestress_result_step",
    "csm_result_step",
    "trim_split_origin",
    "trim_split_normal",
    "trim_floor_origin",
    "trim_floor_normal",
)

_VECTOR_FIELDS = (
    "mri_translation",
    "mri_rotation_deg",
    "trim_split_origin",
    "trim_split_normal",
    "trim_floor_origin",
    "trim_floor_normal",
)


def contains_placeholder(value):
    """Return whether a template placeholder remains anywhere in *value*."""
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return not stripped or (stripped.startswith("<") and stripped.endswith(">"))
    if isinstance(value, Mapping):
        return any(contains_placeholder(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(contains_placeholder(item) for item in value)
    return False


def _require_vector(case_label, field, value):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"case {case_label!r}: {field} must contain three numbers")
    if not all(isinstance(item, Real) for item in value):
        raise ValueError(f"case {case_label!r}: {field} must contain three numbers")


def validate_case(case_label, values):
    """Validate and copy one private case mapping.

    Validation fails before paths are constructed or numerical notebooks start,
    preventing an unedited public template from being mistaken for real input.
    """
    if not isinstance(values, Mapping):
        raise TypeError(f"case {case_label!r} must be a mapping")

    missing = [field for field in REQUIRED_CASE_FIELDS if field not in values]
    if missing:
        raise ValueError(f"case {case_label!r} is missing fields: {', '.join(missing)}")

    unresolved = [field for field in REQUIRED_CASE_FIELDS if contains_placeholder(values[field])]
    if unresolved:
        raise ValueError(
            f"case {case_label!r} still contains template values: {', '.join(unresolved)}"
        )

    if not isinstance(values["heart_rate_bpm"], Real) or values["heart_rate_bpm"] <= 0:
        raise ValueError(f"case {case_label!r}: heart_rate_bpm must be positive")
    for field in ("systolic_mmHg", "diastolic_mmHg"):
        if not isinstance(values[field], Real) or values[field] <= 0:
            raise ValueError(f"case {case_label!r}: {field} must be positive")

    flows = values["outlet_mean_flow"]
    branches = values["outlet_branch_order"]
    if not isinstance(flows, Sequence) or isinstance(flows, (str, bytes)):
        raise ValueError(f"case {case_label!r}: outlet_mean_flow must be a sequence")
    if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes)):
        raise ValueError(f"case {case_label!r}: outlet_branch_order must be a sequence")
    if len(flows) != len(branches) or not flows:
        raise ValueError(
            f"case {case_label!r}: outlet flows and branch names must have equal non-zero length"
        )
    if not all(isinstance(flow, Real) and flow > 0 for flow in flows):
        raise ValueError(f"case {case_label!r}: outlet flows must be positive numbers")
    if len(set(branches)) != len(branches):
        raise ValueError(f"case {case_label!r}: outlet branch names must be unique")

    rings = values["rings_ignored"]
    if not isinstance(rings, Sequence) or isinstance(rings, (str, bytes)):
        raise ValueError(f"case {case_label!r}: rings_ignored must be a sequence")
    if not all(isinstance(index, int) and index >= 0 for index in rings):
        raise ValueError(f"case {case_label!r}: rings_ignored must contain non-negative integers")

    for field in _VECTOR_FIELDS:
        _require_vector(case_label, field, values[field])

    sections = values["mri_outlet_sections"]
    if not isinstance(sections, Mapping) or set(sections) != {"out1", "out2", "out3"}:
        raise ValueError(
            f"case {case_label!r}: mri_outlet_sections must define out1, out2 and out3"
        )
    for name, section in sections.items():
        if not isinstance(section, Mapping):
            raise ValueError(f"case {case_label!r}: {name} section must be a mapping")
        index = section.get("centerline_index")
        radius = section.get("radius_mm")
        if not isinstance(index, int) or index < 0:
            raise ValueError(
                f"case {case_label!r}: {name} centerline_index must be a non-negative integer"
            )
        if not isinstance(radius, Real) or radius <= 0:
            raise ValueError(f"case {case_label!r}: {name} radius_mm must be positive")

    if values["wall_cycle_closure"] != "replace_last_with_reference":
        raise ValueError(
            f"case {case_label!r}: wall_cycle_closure must be "
            "'replace_last_with_reference' for the released Stage-2 implementation"
        )

    for field in (
        "stiffness_target_phase_index",
        "prestress_result_step",
        "csm_result_step",
    ):
        value = values[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"case {case_label!r}: {field} must be a non-negative integer")

    return dict(values)
