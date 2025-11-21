#!/usr/bin/env python3
"""Quick validation test for Euler angle extraction."""

import numpy as np

def test_euler_extraction():
    """Test that a neutral head pose gives angles near 0."""

    # Simulate a nearly neutral rotation matrix (small deviations)
    rotation_mat = np.array([
        [0.998, -0.017, 0.052],
        [0.020,  0.998, -0.035],
        [-0.051, 0.037, 0.998]
    ])

    # Extract Euler angles using the CORRECTED method
    sin_pitch = -rotation_mat[2, 0]
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    cos_pitch = np.cos(pitch)

    if abs(cos_pitch) > 1e-6:
        yaw = np.arctan2(rotation_mat[1, 0] / cos_pitch, rotation_mat[0, 0] / cos_pitch)
        roll = np.arctan2(rotation_mat[2, 1] / cos_pitch, rotation_mat[2, 2] / cos_pitch)
    else:
        yaw = 0
        roll = np.arctan2(-rotation_mat[0, 1], rotation_mat[1, 1])

    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    print("Test: Nearly neutral head pose")
    print(f"  Roll:  {roll_deg:7.2f}° (expected: ~3°)")
    print(f"  Pitch: {pitch_deg:7.2f}° (expected: ~3°)")
    print(f"  Yaw:   {yaw_deg:7.2f}° (expected: ~1°)")

    # Check if angles are reasonable (should be < 10° for neutral pose)
    if abs(roll_deg) < 10 and abs(pitch_deg) < 10 and abs(yaw_deg) < 10:
        print("\n✓ PASS: Angles are in reasonable range for neutral pose")
        return True
    else:
        print("\n✗ FAIL: Angles are too large for neutral pose")
        return False

if __name__ == "__main__":
    success = test_euler_extraction()
    exit(0 if success else 1)

