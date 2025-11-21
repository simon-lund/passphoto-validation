#!/usr/bin/env python3
"""Test script for head pose detection."""

import cv2
import sys
from edutap.image_api.core import PassportPhotoValidator

def test_image(image_path):
    print(f"Testing: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    print(f"Image shape: {image.shape}")

    # Validate
    try:
        validator = PassportPhotoValidator(image, min_detection_confidence=0.5)
        report = validator.validate()

        # Show head pose measurements
        if 'head_roll_degrees' in report.measurements:
            roll = report.measurements['head_roll_degrees']
            pitch = report.measurements['head_pitch_degrees']
            yaw = report.measurements['head_yaw_degrees']

            print(f"\nHead Pose Angles:")
            print(f"  Roll:  {roll:7.2f}° (tilt left/right)")
            print(f"  Pitch: {pitch:7.2f}° (tilt up/down)")
            print(f"  Yaw:   {yaw:7.2f}° (rotation left/right)")

            # Check if angles are reasonable
            if abs(roll) < 10 and abs(pitch) < 10 and abs(yaw) < 10:
                print("\n✓ Head pose looks good!")
            else:
                print("\n✗ Head pose issues detected:")
                if abs(roll) >= 10:
                    print(f"  - Roll angle too large: {abs(roll):.1f}°")
                if abs(pitch) >= 10:
                    print(f"  - Pitch angle too large: {abs(pitch):.1f}°")
                if abs(yaw) >= 10:
                    print(f"  - Yaw angle too large: {abs(yaw):.1f}°")
        else:
            print("\nNo head pose measurements found (face detection may have failed)")

        # Show validation issues related to head pose
        head_issues = [i for i in report.issues if 'head' in i.message.lower() or 'tilt' in i.message.lower() or 'rotat' in i.message.lower()]
        if head_issues:
            print(f"\nValidation Issues ({len(head_issues)}):")
            for issue in head_issues:
                print(f"  [{issue.category.upper()}] {issue.message}")

    except Exception as e:
        print(f"ERROR during validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image(sys.argv[1])
    else:
        # Test with default images
        test_image("tests/test-images/8.jpg")

