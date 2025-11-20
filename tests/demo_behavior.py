# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Command-line demo showcasing comfort model mapping from perception inputs to Spot behaviors
# Acknowledgements: Claude for demo implementation

"""
Command-line helper to demonstrate how the comfort model maps perception inputs
to Spot behaviors.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Optional

from behavior_planner import (
    BehaviorLabel,
    ComfortModel,
    ComfortTuner,
    PerceptionInput,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo tool for Friendly Spot's behavior planner."
    )
    parser.add_argument(
        "--current-action",
        default="moving",
        help="Current Spot action/state (default: moving)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Distance to the person in meters (default: None)",
    )
    parser.add_argument(
        "--face-label",
        default="unknown",
        help="Face label returned by recognition (default: unknown)",
    )
    parser.add_argument(
        "--emotion-label",
        default="neutral",
        help="Emotion label from DeepFace (default: neutral)",
    )
    parser.add_argument(
        "--pose-label",
        default="standing",
        help="Pose/action label from pose estimator (default: standing)",
    )
    parser.add_argument(
        "--gesture-label",
        default="none",
        help="Gesture label from Mediapipe hands (default: none)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run through a canned set of scenarios instead of a single input.",
    )
    parser.add_argument(
        "--tune-demo",
        action="store_true",
        help="Show how feedback automatically adjusts comfort hyperparameters.",
    )
    return parser.parse_args()


def run_single(
    comfort_model: ComfortModel,
    current_action: str,
    distance: Optional[float],
    face_label: str,
    emotion_label: str,
    pose_label: str,
    gesture_label: str,
) -> BehaviorLabel:
    perception = PerceptionInput(
        current_action=current_action,
        distance_m=distance,
        face_label=face_label,
        emotion_label=emotion_label,
        pose_label=pose_label,
        gesture_label=gesture_label,
    )
    comfort, behavior = comfort_model.predict_behavior(perception)
    print(
        f"comfort={comfort:0.2f}  behavior={behavior.value:>18}  "
        f"(action={current_action}, distance={distance}, face={face_label}, "
        f"emotion={emotion_label}, pose={pose_label}, gesture={gesture_label})"
    )
    return behavior


def run_demo(model: ComfortModel) -> None:
    """
    Walk through multiple scenarios to highlight how comfort behaves.
    """

    scenarios: Iterable[PerceptionInput] = [
        PerceptionInput(
            current_action="waiting",
            distance_m=1.1,
            face_label="sally",
            emotion_label="happy",
            pose_label="waving",
            gesture_label="thumbs_up",
        ),
        PerceptionInput(
            current_action="moving",
            distance_m=0.6,
            face_label="unknown",
            emotion_label="angry",
            pose_label="arms_crossed",
            gesture_label="closed_fist",
        ),
        PerceptionInput(
            current_action="moving",
            distance_m=1.5,
            face_label="unknown",
            emotion_label="neutral",
            pose_label="standing",
            gesture_label="open_hand",
        ),
        PerceptionInput(
            current_action="interacting",
            distance_m=0.9,
            face_label="matteo",
            emotion_label="calm",
            pose_label="standing",
            gesture_label="none",
        ),
    ]

    print("Running canned comfort scenarios...\n")
    for perception in scenarios:
        run_single(
            model,
            perception.current_action,
            perception.distance_m,
            perception.face_label,
            perception.emotion_label,
            perception.pose_label,
            perception.gesture_label,
        )


def run_tuning_demo(model: ComfortModel) -> None:
    """
    Demonstrate online tuning: reinforce a desired behavior several times and
    show how the predicted behavior shifts.
    """

    tuner = ComfortTuner(model, learning_rate=0.08)
    perception = PerceptionInput(
        current_action="moving",
        distance_m=0.7,
        face_label="unknown",
        emotion_label="neutral",
        pose_label="arms_crossed",
        gesture_label="closed_fist",
    )
    target_behavior = BehaviorLabel.BACK_AWAY

    print("Tuning demo (reinforcing BACK_AWAY for the same perception):\n")
    for iteration in range(5):
        comfort, behavior = model.predict_behavior(perception)
        print(f"iteration={iteration}  comfort={comfort:0.2f}  behavior={behavior.value}")
        tuner.register_feedback(perception, target_behavior, correct=True)

    print("\nNow assume operator flags BACK_AWAY as wrong and prefers SIT:\n")
    target_behavior = BehaviorLabel.SIT
    for iteration in range(5, 8):
        comfort, behavior = model.predict_behavior(perception)
        print(f"iteration={iteration}  comfort={comfort:0.2f}  behavior={behavior.value}")
        tuner.register_feedback(perception, target_behavior, correct=True)


def main() -> None:
    args = parse_args()
    model = ComfortModel()

    if args.tune_demo:
        run_tuning_demo(model)
    elif args.demo:
        run_demo(model)
    else:
        run_single(
            model,
            args.current_action,
            args.distance,
            args.face_label,
            args.emotion_label,
            args.pose_label,
            args.gesture_label,
        )


if __name__ == "__main__":
    main()
