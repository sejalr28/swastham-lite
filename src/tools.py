

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, Any



def bedtime_calculator(wake_time: str, cycles: int = 5) -> Dict[str, Any]:
    """
    Suggests bedtimes based on 90-minute sleep cycles counted backward from
    a target wake-up time, plus ~15 minutes to fall asleep. Grounded in the
    sleep-cycle info in sh-002 (Sleep Cycles and Stages).

    Args:
        wake_time: 24h "HH:MM" target wake-up time, e.g. "07:00"
        cycles: number of 90-minute cycles to sleep through (4, 5, or 6 typical)
    """
    try:
        wake_dt = datetime.strptime(wake_time, "%H:%M")
    except ValueError:
        return {"error": f"Could not parse wake_time '{wake_time}'. Use 24h HH:MM format."}

    if cycles not in (4, 5, 6):
        return {"error": "cycles should be 4, 5, or 6 (typical full-cycle counts)."}

    fall_asleep_buffer = timedelta(minutes=15)
    cycle_length = timedelta(minutes=90)
    total_sleep = cycle_length * cycles

    bedtime_dt = wake_dt - total_sleep - fall_asleep_buffer
    bedtime_str = bedtime_dt.strftime("%H:%M")

    return {
        "wake_time": wake_time,
        "recommended_bedtime": bedtime_str,
        "cycles": cycles,
        "total_sleep_hours": round(total_sleep.total_seconds() / 3600, 1),
        "note": ("Based on ~90-minute sleep cycles plus ~15 minutes to fall asleep. "
                 "This is a general estimate, not personalized medical guidance - "
                 "actual cycle length and time to fall asleep vary by individual."),
    }


BEDTIME_CALCULATOR_PARAMS = {
    "type": "object",
    "properties": {
        "wake_time": {"type": "string", "description": "24h HH:MM target wake time, e.g. '07:00'"},
        "cycles": {"type": "integer", "description": "Number of 90-min sleep cycles (4, 5, or 6)", "default": 5},
    },
    "required": ["wake_time"],
}




def caffeine_cutoff_calculator(bedtime: str, sensitivity: str = "average") -> Dict[str, Any]:
    """
    Suggests a last-safe-caffeine time based on bedtime, using the
    6-8 hour half-life guidance in sh-006 (Caffeine, Alcohol, and Sleep
    Quality). Not medical dosing advice -- purely a scheduling suggestion.

    Args:
        bedtime: 24h "HH:MM" intended bedtime
        sensitivity: "average" (6h cutoff) or "high" (8h cutoff) caffeine sensitivity
    """
    try:
        bed_dt = datetime.strptime(bedtime, "%H:%M")
    except ValueError:
        return {"error": f"Could not parse bedtime '{bedtime}'. Use 24h HH:MM format."}

    hours_map = {"average": 6, "high": 8}
    if sensitivity not in hours_map:
        return {"error": "sensitivity should be 'average' or 'high'."}

    cutoff_hours = hours_map[sensitivity]
    cutoff_dt = bed_dt - timedelta(hours=cutoff_hours)
    cutoff_str = cutoff_dt.strftime("%H:%M")

    return {
        "bedtime": bedtime,
        "sensitivity": sensitivity,
        "suggested_caffeine_cutoff": cutoff_str,
        "note": (f"Based on a general {cutoff_hours}-hour caffeine half-life guideline. "
                 "Individual metabolism varies; this is a scheduling suggestion, not "
                 "medical advice."),
    }


CAFFEINE_CUTOFF_PARAMS = {
    "type": "object",
    "properties": {
        "bedtime": {"type": "string", "description": "24h HH:MM intended bedtime, e.g. '22:30'"},
        "sensitivity": {"type": "string", "enum": ["average", "high"], "default": "average"},
    },
    "required": ["bedtime"],
}


# ---------------------------------------------------------------------------
# Tool 3: Wind-down routine builder (based on sh-009)
# ---------------------------------------------------------------------------

_ROUTINE_STEPS = [
    ("Dim the lights / switch devices to night mode", 5),
    ("Write down tomorrow's to-do list or any worries", 5),
    ("Light stretching or a warm shower", 10),
    ("Slow breathing exercise (e.g. 4-7-8 breathing)", 5),
    ("Reading or quiet activity in low light", 15),
]


def build_winddown_routine(bedtime: str, duration_minutes: int = 40) -> Dict[str, Any]:
    """
    Builds a scheduled wind-down routine ending at bedtime, using the
    general stress-reduction strategies in sh-009 (Stress, Racing Thoughts,
    and Sleep). Steps and durations are illustrative, not prescriptive.

    Args:
        bedtime: 24h "HH:MM" intended bedtime
        duration_minutes: total wind-down window length (default 40)
    """
    try:
        bed_dt = datetime.strptime(bedtime, "%H:%M")
    except ValueError:
        return {"error": f"Could not parse bedtime '{bedtime}'. Use 24h HH:MM format."}

    if duration_minutes < 10:
        return {"error": "duration_minutes should be at least 10 to fit a meaningful routine."}

    # Scale the default step durations to fit the requested window
    default_total = sum(d for _, d in _ROUTINE_STEPS)
    scale = duration_minutes / default_total

    scaled_steps = [(name, max(1, round(d * scale))) for name, d in _ROUTINE_STEPS]
    # Adjust rounding drift so total exactly matches duration_minutes
    drift = duration_minutes - sum(d for _, d in scaled_steps)
    if scaled_steps:
        last_name, last_dur = scaled_steps[-1]
        scaled_steps[-1] = (last_name, last_dur + drift)

    start_dt = bed_dt - timedelta(minutes=duration_minutes)
    schedule = []
    cursor = start_dt
    for name, dur in scaled_steps:
        schedule.append({
            "time": cursor.strftime("%H:%M"),
            "duration_minutes": dur,
            "activity": name,
        })
        cursor += timedelta(minutes=dur)

    return {
        "bedtime": bedtime,
        "routine_start": start_dt.strftime("%H:%M"),
        "total_duration_minutes": duration_minutes,
        "schedule": schedule,
        "note": ("A general wind-down template based on common relaxation strategies. "
                 "Adjust activities to personal preference."),
    }


WINDDOWN_ROUTINE_PARAMS = {
    "type": "object",
    "properties": {
        "bedtime": {"type": "string", "description": "24h HH:MM intended bedtime, e.g. '22:30'"},
        "duration_minutes": {"type": "integer", "description": "Total wind-down window in minutes", "default": 40},
    },
    "required": ["bedtime"],
}


# ---------------------------------------------------------------------------
# Registry: what agent.py routes over
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "bedtime_calculator": {
        "fn": bedtime_calculator,
        "description": bedtime_calculator.__doc__.strip().split("\n")[0],
        "parameters": BEDTIME_CALCULATOR_PARAMS,
       
        "pattern_strings": [
            r"what time (should|do) i (go to bed|sleep)",
            r"wake up at",
            r"wake[- ]?time",
            r"sleep cycles",
        ],
    },
    "caffeine_cutoff_calculator": {
        "fn": caffeine_cutoff_calculator,
        "description": caffeine_cutoff_calculator.__doc__.strip().split("\n")[0],
        "parameters": CAFFEINE_CUTOFF_PARAMS,
        "pattern_strings": [
            r"last (cup of coffee|coffee|time i should have caffeine|caffeine)",
            r"caffeine cutoff",
            r"when should i stop (drinking coffee|drinking caffeine|having caffeine)",
            r"stop drinking caffeine",
        ],
    },
    "build_winddown_routine": {
        "fn": build_winddown_routine,
        "description": build_winddown_routine.__doc__.strip().split("\n")[0],
        "parameters": WINDDOWN_ROUTINE_PARAMS,
        "pattern_strings": [
            r"wind[- ]?down routine",
            r"wind[- ]?down",
            r"routine before bed",
            r"bedtime routine",
        ],
    },
}


if __name__ == "__main__":
    print(bedtime_calculator("07:00"))
    print(caffeine_cutoff_calculator("22:30", sensitivity="high"))
    print(build_winddown_routine("23:00", duration_minutes=30))
