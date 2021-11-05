# README
## Data Preparation
1. Use `pip install -r requirements.txt` to download required packages.
2. Download 3DPW dataset from https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html, and put it under `data/3DPW/` folder.
3. Preprocess 3DPW data using `python scripts/preprocess_3DPW.py`
4. Download all the https://amass.is.tue.mpg.de/index.html, and extract them to `data/AMASS/` folder.
5. Preprocess AMASS data using `python scripts/preprocess_AMASS.py`
6. Download SMPL model and place it under `data/smplx_models/smpl/`

If everything setups properly, the layout of `data/` folder will be something like:
```
 data
 ├── 3DPW
 │   ├── imageFiles/
 │   └── sequenceFiles/
 ├── AMASS
 │    ├── ACCAD/
 │    ├── BioMotionLab_NTroje/
 │    ├── BMLhandball/
 │    ├── BMLmovi/
 │    ├── CMU/
 │    ├── DanceDB/
 │    ├── DFaust_67/
 │    ├── EKUT/
 │    ├── Eyes_Japan_Dataset/
 │    ├── HUMAN4D/
 │    ├── HumanEva/
 │    ├── KIT/
 │    ├── MPI_HDM05/
 │    ├── MPI_Limits/
 │    ├── MPI_mosh/
 │    ├── SFU/
 │    ├── SSM_synced/
 │    ├── TCD_handMocap/
 │    ├── TotalCapture/
 │    └── Transitions_mocap/
 ├── 3DPW_test.npz
 ├── 3DPW_valid.npz
 ├── AMASS.npz
 ├── J_regressor_h36m.npy
 └── smplx_models
     └── smpl
         ├── SMPL_FEMALE.pkl
         ├── SMPL_MALE.pkl
         └── SMPL_NEUTRAL.pkl
```

## Training & Evaluation
```
# Evaluate with sample pretrained model
python eval.py

# (Optional) train from scratch
python train.py
```
