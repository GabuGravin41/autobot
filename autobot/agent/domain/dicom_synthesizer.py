"""
DICOM Medical Imaging & 3D Volume Synthesizer — Generates pydicom scripts to compute cranium volume from CT scan slices.
"""
import logging

logger = logging.getLogger(__name__)


class DICOMVolumeSynthesizer:
    """
    Synthesizes pydicom / numpy scripts to parse CT DICOM slice stacks,
    extract voxel spacing and slice thickness, apply HU thresholds, and calculate 3D volume.
    """

    @staticmethod
    def generate_volume_script(dicom_dir: str = "dicom_slices", hu_min_threshold: int = 300, output_json: str = "cranium_volume.json") -> str:
        """
        Generates a standalone Python script to compute 3D volume from a folder of DICOM files.
        """
        # repr() for strings, int() for the threshold — not raw interpolation
        # into a literal. The previous version used `r"{dicom_dir}"`, and a
        # raw string still terminates on an unescaped quote character, so a
        # dicom_dir/output_json containing `"` broke out of the literal in
        # the GENERATED script. hu_min_threshold is typed as int but nothing
        # enforced that at runtime; int() here does, so a non-numeric value
        # fails loudly at generation time instead of injecting arbitrary text
        # into the script as a bare (unquoted) expression.
        dicom_dir_lit = repr(dicom_dir)
        output_json_lit = repr(output_json)
        hu_threshold_val = int(hu_min_threshold)
        return f'''# Auto-generated DICOM 3D Volume Calculation Script
import os
import sys
import glob
import json

try:
    import pydicom
    import numpy as np
except ImportError:
    print("⚠️ pydicom or numpy not found. Install with 'pip install pydicom numpy'.")
    sys.exit(1)

dicom_dir = {dicom_dir_lit}
hu_threshold = {hu_threshold_val}
output_json = {output_json_lit}

print(f"🏥 Scanning DICOM slices in {{dicom_dir}} (HU Threshold >= {{hu_threshold}})...")

files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
if not files:
    print(f"⚠️ No DICOM (.dcm) files found in {{dicom_dir}}.")
    sys.exit(0)

# Read slices and sort by InstanceNumber or ImagePositionPatient
slices = [pydicom.dcmread(f) for f in files]
slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if hasattr(x, "ImagePositionPatient") else int(x.InstanceNumber))

# Calculate voxel volume (dx * dy * dz)
pixel_spacing = slices[0].PixelSpacing
slice_thickness = getattr(slices[0], "SliceThickness", 1.0)
voxel_volume_mm3 = float(pixel_spacing[0]) * float(pixel_spacing[1]) * float(slice_thickness)

total_voxels = 0
for s in slices:
    img = s.pixel_array
    # Convert to Hounsfield Units (HU)
    hu_img = img * float(s.RescaleSlope) + float(s.RescaleIntercept) if hasattr(s, "RescaleSlope") else img
    mask = hu_img >= hu_threshold
    total_voxels += np.sum(mask)

total_volume_cm3 = (total_voxels * voxel_volume_mm3) / 1000.0

result = {{
    "slices_count": len(slices),
    "voxel_volume_mm3": voxel_volume_mm3,
    "total_voxels_segmented": int(total_voxels),
    "volume_cm3": round(total_volume_cm3, 2),
}}

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(f"✅ 3D Segmentation Volume: {{total_volume_cm3:.2f}} cm³ (Saved to {{output_json}})")
'''
