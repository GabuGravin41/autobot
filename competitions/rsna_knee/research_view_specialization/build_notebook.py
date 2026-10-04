"""
Builder script for RSNA Knee View-Specialization Probe.
Constructs a self-contained Kaggle notebook that explicitly decouples prediction heads
based on MRI planes (Sagittal, Coronal, Axial).

Plane decoupling:
- Head A (Sagittal): predicts ['ACL', 'Contusion', 'Fracture']
- Head B (Coronal): predicts ['MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA']
- Head C (Axial): predicts ['PF OA', 'Effusion', 'Synovitis', "Baker's"]
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DST_NB = HERE / "notebook.ipynb"

def create_notebook():
    cells = []
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# RSNA Knee: View Specialization Probe\n",
            "\n",
            "This notebook tests an explicit plane decoupling strategy where specific pathologies are predicted exclusively from their optimal anatomical views:\n",
            "- **Model Head A (Sagittal)**: `['ACL', 'Contusion', 'Fracture']`\n",
            "- **Model Head B (Coronal)**: `['MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA']`\n",
            "- **Model Head C (Axial)**: `['PF OA', 'Effusion', 'Synovitis', \"Baker's\"]`\n"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import gc\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "from torch.utils.data import Dataset, DataLoader\n",
            "\n",
            "TARGETS_SAGITTAL = ['ACL', 'Contusion', 'Fracture']\n",
            "TARGETS_CORONAL = ['MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA']\n",
            "TARGETS_AXIAL = ['PF OA', 'Effusion', 'Synovitis', \"Baker's\"]\n",
            "\n",
            "print(\"Sagittal targets:\", TARGETS_SAGITTAL)\n",
            "print(\"Coronal targets:\", TARGETS_CORONAL)\n",
            "print(\"Axial targets:\", TARGETS_AXIAL)\n"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class ViewSpecializedModel(nn.Module):\n",
            "    def __init__(self, backbone_name='resnet18', pretrained=True):\n",
            "        super().__init__()\n",
            "        import timm\n",
            "        # Separate feature extractors per view could be used, or a shared backbone.\n",
            "        # Here we demonstrate separate backbones per plane for complete decoupling.\n",
            "        self.backbone_sagittal = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, in_chans=1)\n",
            "        self.backbone_coronal = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, in_chans=1)\n",
            "        self.backbone_axial = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, in_chans=1)\n",
            "        \n",
            "        embed_dim = self.backbone_sagittal.num_features\n",
            "        \n",
            "        # View-specific heads\n",
            "        self.head_sagittal = nn.Linear(embed_dim, len(TARGETS_SAGITTAL))\n",
            "        self.head_coronal = nn.Linear(embed_dim, len(TARGETS_CORONAL))\n",
            "        self.head_axial = nn.Linear(embed_dim, len(TARGETS_AXIAL))\n",
            "        \n",
            "    def forward(self, x_sag, x_cor, x_ax):\n",
            "        # x_shape: (B, 1, H, W) for a chosen slice, or 3D\n",
            "        f_sag = self.backbone_sagittal(x_sag)\n",
            "        f_cor = self.backbone_coronal(x_cor)\n",
            "        f_ax = self.backbone_axial(x_ax)\n",
            "        \n",
            "        out_sag = self.head_sagittal(f_sag)\n",
            "        out_cor = self.head_coronal(f_cor)\n",
            "        out_ax = self.head_axial(f_ax)\n",
            "        \n",
            "        # Return a dictionary of predictions\n",
            "        return {\n",
            "            'sagittal': out_sag,\n",
            "            'coronal': out_cor,\n",
            "            'axial': out_ax\n",
            "        }\n",
            "\n",
            "model = ViewSpecializedModel(pretrained=False)\n",
            "print(\"Model instantiated.\")\n"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Dummy data validation loop to verify output schema\n",
            "def validate_schema():\n",
            "    batch_size = 4\n",
            "    dummy_sag = torch.randn(batch_size, 1, 224, 224)\n",
            "    dummy_cor = torch.randn(batch_size, 1, 224, 224)\n",
            "    dummy_ax = torch.randn(batch_size, 1, 224, 224)\n",
            "    \n",
            "    model.eval()\n",
            "    with torch.no_grad():\n",
            "        preds = model(dummy_sag, dummy_cor, dummy_ax)\n",
            "    \n",
            "    assert preds['sagittal'].shape == (batch_size, 3)\n",
            "    assert preds['coronal'].shape == (batch_size, 5)\n",
            "    assert preds['axial'].shape == (batch_size, 4)\n",
            "    print(\"Validation successful. Decoupled heads returned correct target dimensions.\")\n",
            "    \n",
            "validate_schema()\n"
        ]
    })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    DST_NB.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Generated {DST_NB.name}")

if __name__ == "__main__":
    create_notebook()
