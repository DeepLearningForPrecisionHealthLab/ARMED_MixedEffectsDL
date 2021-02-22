#!/bin/bash
# Converts DICOM Seg files in the QIN-Prostate dataset into NIfTI files using dcmqi. 
# Run this script using the singularity_qin_segmentations.sh wrapper. 
DICOMDIR=/archive/bioinformatics/DLLab/STUDIES/Prostate_Multi_Datasets_20210209/QIN-Prostate/dicom
NIFTIDIR=/archive/bioinformatics/DLLab/STUDIES/Prostate_Multi_Datasets_20210209/QIN-Prostate/nifti

# Prevent the for loop from splitting paths with spaces
SAFEIFS=$IFS
IFS=$(echo -en "\n\b")

if [ ! -d $NIFTIDIR ]; then
    mkdir -p $NIFTIDIR
fi

# Loop through segmentation files
for dicom in $(ls $DICOMDIR/*/*/*Segmentations*/*.dcm); do
    filename=$(basename "$dicom")
    dir=$(dirname "$dicom")
    dir=$(dirname "$dir") 
    session=$(basename "$dir")
    dir=$(dirname "$dir")
    subject=$(basename "$dir")

    outdir=$NIFTIDIR/$subject/$session
    echo "$outdir"
    
    if [ ! -d $outdir ]; then
        mkdir -p $outdir
    fi

    segimage2itkimage -t nifti -p segmentation --outputDirectory $outdir --inputDICOM $dicom
done
IFS=$SAVEIFS