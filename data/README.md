# Dataset setup

ChestXpert uses the **NIH ChestX-ray14** research dataset for the first training experiment. It contains 112,120 frontal chest X-rays from 30,805 patients and image-level labels for 14 thoracic findings. The project does not commit the dataset to Git.

Place the downloaded research data locally as:

```
data/raw/NIH_ChestXray14/
├── Data_Entry_2017_v2020.csv
├── train_val_list.txt
├── test_list.txt
└── images/
    ├── 00000001_000.png
    └── ...
```

Then run:

```bash
python scripts/prepare_metadata.py
```

The repository intentionally does not bundle images or patient-level data.

The first experiment uses the dataset's 14 image-level findings. The model will be evaluated on a held-out test split, and reported metrics will state the exact split and preprocessing used.

**Research-only:** follow the dataset provider's terms and do not use this project as a clinical diagnostic system.
