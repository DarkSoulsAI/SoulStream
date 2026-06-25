import os
import sys
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from bonfire_palm import BonfirePalmController


@dataclass
class FakeHand:
    detected: bool
    is_open_palm: bool
    palm_ndc_x: float = 0.12
    palm_ndc_y: float = -0.08


class FakeParticles:
    def __init__(self):
        self.kindle_calls = []
        self.spark_calls = []

    def kindle_nearby(self, x, y, radius=0.3):
        self.kindle_calls.append((round(x, 3), round(y, 3), round(radius, 3)))

    def spawn_palm_sparks(self, x, y, count=15, spread=0.03, intensity=1.0):
        self.spark_calls.append((
            round(x, 3),
            round(y, 3),
            int(count),
            round(spread, 3),
            round(intensity, 3),
        ))


def _hand_for(label):
    if label == "none":
        return None
    if label == "closed":
        return FakeHand(detected=True, is_open_palm=False)
    if label == "open":
        return FakeHand(detected=True, is_open_palm=True)
    raise ValueError(f"Unknown label: {label}")


def run_sequence():
    ctrl = BonfirePalmController()
    particles = FakeParticles()
    dt = 1.0 / 10.0

    sequence = [
        ("none", 5),
        ("closed", 5),
        ("open", 14),
        ("closed", 8),
        ("none", 6),
    ]

    rows = []
    frame = 0
    for label, count in sequence:
        for _ in range(count):
            frame += 1
            state = ctrl.update(_hand_for(label), dt, particles=particles)
            rows.append({
                "frame": frame,
                "input": label,
                "phase": state.phase,
                "charge": state.charge,
                "radius": state.kindle_radius,
                "sun": state.sun_sync,
                "ignited": state.just_ignited,
                "sparks": len(particles.spark_calls),
            })
    return rows, particles


def main():
    rows, particles = run_sequence()

    print("frame input   phase             charge radius sun  ignited sparks")
    print("----- ------- ----------------- ------ ------ ---- ------- ------")
    for row in rows:
        print(
            f"{row['frame']:>5} "
            f"{row['input']:<7} "
            f"{row['phase']:<17} "
            f"{row['charge']:.2f}   "
            f"{row['radius']:.2f}   "
            f"{row['sun']:.2f} "
            f"{str(row['ignited']):<7} "
            f"{row['sparks']:>6}"
        )

    last_spark = particles.spark_calls[-1] if particles.spark_calls else None
    print()
    print(f"kindle_calls={len(particles.kindle_calls)}")
    print(f"spark_calls={len(particles.spark_calls)}")
    print(f"last_spark={last_spark}")


if __name__ == "__main__":
    main()
