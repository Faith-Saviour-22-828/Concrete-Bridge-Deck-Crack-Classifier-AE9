# Concrete Bridge Deck Crack Detection

Binary image classification app (Cracked vs Non-cracked) built for **GET 324 — Laboratory
Exercise 10 (Mini-Project): Cloud Computing and AI Model Deployment for Engineering Applications**.

## Problem statement
Bridge decks and other concrete structures develop surface cracks over time. This project trains
a Convolutional Neural Network to automatically classify an image of a concrete surface as
**Cracked** or **Non-cracked**, then deploys the model as a Streamlit web application.

## Dataset
"Concrete Crack Images for Classification" — Ozgenel, C.F. & Gonenc Sorguc, A. (2018),
Mendeley Data, V2, DOI: 10.17632/5y9wdsg2zt.2 (also mirrored on Kaggle as
`arunrk7/surface-crack-detection`). 40,000 real photographs (227x227 px) of concrete surfaces
from METU campus buildings, evenly split between cracked and non-cracked classes. No synthetic
data was used.

## Models trained
1. **Custom CNN** — 3 convolutional blocks with batch normalisation and max-pooling, trained from
   scratch.
2. **MobileNetV2 (transfer learning)** — ImageNet-pretrained backbone, frozen then fine-tuned.

Both models are trained, evaluated, and compared inside
`notebooks/GET324_Crack_Detection_MiniProject.ipynb` using accuracy, precision, recall, a
confusion matrix, and ROC-AUC on a held-out test split. The better-performing model is exported
to `model/crack_detector.keras` and used by the app.

## Repository structure
```
concrete-crack-detector/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md
├── model/
│   └── crack_detector.keras
└── notebooks/
    └── GET324_Crack_Detection_MiniProject.ipynb
```

## Running locally
```bash
git clone https://github.com/<your-username>/concrete-crack-detector.git
cd concrete-crack-detector
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## Using the app
1. Open the deployed app URL (or the local URL above).
2. Upload a photo of a concrete surface (`.jpg`, `.jpeg`, or `.png`).
3. The app displays the predicted class (**Cracked** / **Non-cracked**) and a confidence score.

## Deployment
Deployed on **Streamlit Community Cloud**: `https://<your-app-name>.streamlit.app`
(replace with your actual deployed URL before submission).

## Team
See the notebook header for the full list of team members, registration numbers, and individual
contributions.

| Name | Registration Number | GitHub Username |
|---|---|---|
| FAITH BEN SAVIOUR | 22/EG/AE/828 | Faith-Saviour-22-828 |
| UDOM UDEME UDEME | 22/EG/AE/778 | Udom-Udeme-22-778 |
| WILLIAM GOD'SWILL EDET  | 23/EG/AE/038 | Godswill-William-23-038 |
| AKPAN EKEMINI MFON | 22/EG/AE/808 | Ekemini-Akpan-22-808 |
| EKERETTE NDIKPONKE ANIEFIOK | 23/EG/AE/018 | Ndikponke-Ekerette-23-018 |
| OKON DONALD EDET | 22/EG/AE/788  | Donald-Okon-22-788 |
| BENJAMIN RICHARD OFFIONG  | 23/EG/AE/008 | Richard- Benjamin-23-008 |
| BASSEY ISRAEL ESSIEN  | 22/EG/AE/798 | Israel-Bassey-22-798 |
| GIDEON EDIMA GEORGE | 23/EG/AE/028 | Gideon-George-23-028 |


## Citation
Ozgenel, C.F. (2019), "Concrete Crack Images for Classification", Mendeley Data, V2,
doi: 10.17632/5y9wdsg2zt.2. Licensed under CC BY 4.0.
