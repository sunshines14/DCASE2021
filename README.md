# DCASE2021
This repository is the official implementations of our DCASE 2021 task 1a with technical report:

Soonshin Seo, Ji-Hwan Kim: "[MobileNet using Coordinate Attention and Fusions for Low-Complexity Acoustic Scene Classification with Multiple Devices](http://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Seo_52_t1.pdf)", submitted to task 1a of the 2021 DCASE Challenge 
  
## Training
 1. Download the development dataset form links at https://zenodo.org/record/3819968#.YLiqhfkzaUk
 2. Use the script "feats.py" & data augmentation scripts
 3. Use the script "train.py"
	  
## Quantization & Evauation
 1. Use the script "run.sh"
 
## Main methods
 1. Normalization & data augmentations
 2. MobileNet
 3. Cooridnate attention 
 4. Early fusion & late fusion
		 
## Acknowledgement
We used the implementation presented in https://github.com/MihawkHu/DCASE2020_task1 as our baseline script.

## Bibtex
```
@techreport{Seo_DCASE2021,
  author    = {Soonshin Seo, Ji-Hwan Kim},
  title     = {MobileNet using Coordinate Attention and Fusions for Low-Complexity Acoustic Scene Classification with Multiple Devices},
  institution = {DCASE2021 Challenge},
  year      = {2021},
}
```
