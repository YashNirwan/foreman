"""The hazard taxonomy the whole pipeline is scored against.

Keeping this in one place is what makes the eval numbers mean anything. The
perception prompt, the verifier's reject criteria, the ground-truth label schema and
the UI filters all read from here, so a class cannot drift between the model that
proposes it and the harness that grades it.

Classes were chosen from the OSHA powered industrial truck standard (29 CFR 1910.178)
and the struck-by categories that dominate warehouse injury statistics, not from
what a VLM happens to be good at describing. That ordering matters: the taxonomy is
a claim about what an inspector needs, and the model has to meet it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HazardClass:
    key: str
    label: str
    definition: str
    standard: str
    # What must be visible for this to be a real detection. The verifier quotes these
    # back when rejecting, which is what keeps rejections auditable.
    requires: str
    # The failure mode this class attracts. Written from observed model behaviour on
    # the eval set, not guessed.
    common_false_positive: str


HAZARDS: list[HazardClass] = [
    HazardClass(
        key="pedestrian_in_path",
        label="Pedestrian in truck path",
        definition=(
            "A person on foot is in, or is walking into, the travel path of a powered "
            "industrial truck that is moving or about to move."
        ),
        standard="29 CFR 1910.178(n)(1) - operator shall slow or stop for pedestrians",
        requires=(
            "Both a person on foot and a powered industrial truck visible in the same "
            "frame, with the person in the truck's direction of travel and close enough "
            "that the truck could not stop short of them."
        ),
        common_false_positive=(
            "A worker standing safely beside or behind a parked truck, or an operator "
            "dismounting their own truck, scored as if they were in its path."
        ),
    ),
    HazardClass(
        key="missing_ppe",
        label="Missing PPE",
        definition=(
            "A person in an active operating area without required high-visibility "
            "clothing or head protection."
        ),
        standard="29 CFR 1910.132 - personal protective equipment, general requirements",
        requires=(
            "A clearly visible person, upper body in frame, in an area where powered "
            "equipment is operating, with no hi-vis vest or hard hat present."
        ),
        common_false_positive=(
            "Poor lighting or low resolution reading as absent PPE, or an office and "
            "break area treated as an active operating area."
        ),
    ),
    HazardClass(
        key="unsafe_load",
        label="Unsafe load handling",
        definition=(
            "A load is raised while the truck travels, is visibly unstable or "
            "overhanging, or is carried high enough to obstruct the operator's view."
        ),
        standard="29 CFR 1910.178(o) - truck and load shall be handled safely",
        requires=(
            "A load on the forks, plus evidence of travel with forks elevated, visible "
            "tilt or overhang, or the load blocking the operator's forward sightline."
        ),
        common_false_positive=(
            "Normal lifting at a rack face, where elevation is the job rather than a "
            "hazard, scored as travelling with a raised load."
        ),
    ),
    HazardClass(
        key="blocked_egress",
        label="Blocked egress or equipment",
        definition=(
            "An exit route, marked aisle, electrical panel or fire suppression device "
            "is obstructed by stored material."
        ),
        standard="29 CFR 1910.37(a)(3) - exit routes kept free of obstruction",
        requires=(
            "A visible exit door, exit sign, marked aisle, extinguisher or panel with "
            "material physically in front of it."
        ),
        common_false_positive=(
            "Material staged in a normal storage bay near a wall read as blocking an "
            "exit that is not actually visible in frame."
        ),
    ),
    HazardClass(
        key="unsafe_operation",
        label="Unsafe truck operation",
        definition=(
            "An unauthorised rider, a person under raised forks, an unattended running "
            "truck, or riding on the forks."
        ),
        standard="29 CFR 1910.178(m) - truck operations, unattended and rider rules",
        requires=(
            "A second person on the truck who is not the operator, a person positioned "
            "beneath an elevated load, or an operator more than 25 feet from a running "
            "truck with forks raised."
        ),
        common_false_positive=(
            "A spotter or co-worker walking beside a truck scored as an unauthorised "
            "rider."
        ),
    ),
]

BY_KEY = {h.key: h for h in HAZARDS}
KEYS = [h.key for h in HAZARDS]

SEVERITIES = ["low", "medium", "high"]


def taxonomy_prompt_block() -> str:
    """Render the taxonomy for the perception prompt."""
    lines = []
    for h in HAZARDS:
        lines.append(f"- {h.key}: {h.definition}")
        lines.append(f"    only counts if: {h.requires}")
    return "\n".join(lines)


def verifier_prompt_block() -> str:
    """Render the taxonomy for the verifier, including the known failure modes.

    The verifier gets the false-positive notes and the perception pass does not. That
    asymmetry is deliberate: telling the proposer what not to say suppresses recall,
    while telling the adjudicator what to watch for raises precision. The eval in
    `evals/run_eval.py` is what established that this ordering wins.
    """
    lines = []
    for h in HAZARDS:
        lines.append(f"- {h.key} ({h.label})")
        lines.append(f"    definition: {h.definition}")
        lines.append(f"    evidence required: {h.requires}")
        lines.append(f"    frequent false positive: {h.common_false_positive}")
        lines.append(f"    standard: {h.standard}")
    return "\n".join(lines)
