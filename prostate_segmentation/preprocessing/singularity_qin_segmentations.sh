#!/bin/bash
# Wrapper to run the convert_qin_segmentations.sh script inside the dcmqi Docker container.

# Path to dcmqi singularity image (pulled from docker)
DCMQI=/archive/bioinformatics/DLLab/KevinNguyen/lib/dcmqi/dcmqi.simg
# Input file directory
DICOMDIR=/archive/bioinformatics/DLLab/STUDIES/Prostate_Multi_Datasets_20210209/QIN-Prostate/dicom
# Output directory
NIFTIDIR=/archive/bioinformatics/DLLab/STUDIES/Prostate_Multi_Datasets_20210209/QIN-Prostate/nifti

cat convert_qin_segmentations.sh | singularity exec -B $DICOMDIR -B $NIFTIDIR $DCMQI bash