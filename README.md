# LocaVQG
Public repository for LocaVQG: Location-aware Visual Question Generation with Lightweight Models. The paper is published in EMNLP 2023.
The dataset is collected by the method informed in the paper, we also provided sample prompt to query to GPT-4 using own API Key. 

## Dataset Structure
We split the dataset into train, val, and test set. Each set is formed as a json file with the structure shown below.

```
{
  [Location / Image ID] : {
    "Caption": [ ... ],
    "Question": [ ... ]
    }
}
```

Each Image ID corresponds to the IDS in [Google Street view Dataset](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/)[1]. This data is collected in order to train smaller language model: T5, VL-T5 that are used in MVQG[2].
For each image id, we provided the captions and questions corresponding to the iamge id. The first four entry to the caption is the captions of the street view images from four directions, the last entry of the caption is from the street address of the location. The questions are collected and filtered from GPT-4. we select top-5 questions from 10 questions that are generated.

We also provided the raw questions in the directory

## Model
We also provide our engaging classifier and FDT5 codes. We did not provide the checkpoint, so you will need to train it to test it. Here are the instructions on how to use FDT5

## References
[1] Amir Roshan Zamir and Mubarak Shah (2014), "Image Geo-localization Based on Multiple Nearest Neighbor Feature Matching using Generalized Graphs", IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
[2] Yeh et al (2022), "Multi-VQG: Generating Engaging Questions for Multiple Images", EMNLP 2022.
