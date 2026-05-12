#MA571 Capstone Project: Breast Density Classification
This repository contains the implementation and comparative analysis of NYU Density Classification and LIBRA methodologies. The project utilizes multiview CNNs and histogram-based methods, and applies threshold optimization to classify breast density against clinical BIRAD assessments.

## Project Overview
The project is divided into two primary workflows:
- NYU Density Classification: An automated pipeline for processing DICOM images and predicting density using CNN and Histogram models.
- LIBRA Integration: A process to analyze density percentages obtained from the CaPTk LIBRA tool.

## NYU Density Classification Workflow
To run the NYU density pipeline, the steps are as follows:
1. Data Input: Upload FFDM DICOM images to the directory: `dicom_images/`.
2. Preprocessing: Run `preprocessing_dicom.py`.
   - The script validates metadata tags (laterality and view position) and organizes images into structured parent folders.
3. Model Execution: Run `run_model_density.py`.
   - This applies both the CNN and Histogram methods. Results are exported to density_predictions_summary.csv.
4. Evaluation: `Run kappa.py` to calculate the weighted quadratic kappa. This compares model predictions against clinic-assessed BIRADs.

## LIBRA Workflow
This module requires pre-computed results from the LIBRA CaPTk software.
1. Obtain the density output file from LIBRA and define the path on the `libra_density_model.py`.
2. Run `libra_density_model.py`.
   - This merges LIBRA density percentages with original BIRADs data.
   - Applies threshold optimization to convert continuous percentages into categorical BIRAD groups.
   - Calculated the weighted quadratic Kappa for agreement with clinical assessment.

## LIBRA Reference 
Pati, S. et al. (2020). The Cancer Imaging Phenomics Toolkit (CaPTk): Technical Overview. In: Crimi, A., Bakas, S. (eds) Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. BrainLes 2019. Lecture Notes in Computer Science(), vol 11993. Springer, Cham. https://doi.org/10.1007/978-3-030-46643-5_38

## NYU Reference
**Breast density classification with deep convolutional neural networks**\
Nan Wu, Krzysztof J. Geras, Yiqiu Shen, Jingyi Su, S. Gene Kim, Eric Kim, Stacey Wolfson, Linda Moy, Kyunghyun Cho\
*ICASSP, 2018*

    @inproceedings{breast_density,
        title = {Breast density classification with deep convolutional neural networks},
        author = {Nan Wu and Krzysztof J. Geras and Yiqiu Shen and Jingyi Su and S. Gene Kim and Eric Kim and Stacey Wolfson and Linda Moy and Kyunghyun Cho},
        booktitle = {ICASSP},
        year = {2018}
    }
