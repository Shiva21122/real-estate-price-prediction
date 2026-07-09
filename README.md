# Real Estate Price Prediction

Predicts residential property sale prices from features like square footage, beds/baths, property tax, and property type. Compares Linear Regression, Decision Tree, and Random Forest models, and serves the best one through a Streamlit app.

## Business Question

What is a fair market price for a property given its physical characteristics, location costs, and market timing — and which features drive that price?

## Results

| Model | Test MAE |
|-------|----------|
| Linear Regression | ~$83,500 |
| Decision Tree (depth 3) | ~$67,800 |
| **Random Forest (200 trees)** | **~$45,000** |

Random Forest cut the average pricing error roughly in half vs. the linear baseline and is the deployed model.

## Project Structure

```
real-estate-price-prediction/
├── app.py                      # Streamlit web app
├── train_model.py              # Training pipeline (LR vs DT vs RF)
├── data/
│   └── final.csv               # Cleaned dataset
├── models/
│   └── real_estate_model.pkl   # Deployed Random Forest model
├── assets/
│   └── tree.png                # Decision tree visualization
├── notebooks/
│   └── real_estate_price_model.ipynb
├── requirements.txt
└── README.md
```

## Features Used

year_sold, property_tax, insurance, beds, baths, sqft, year_built, lot_size, basement, popular (location flag), recession (market timing), property_age, property_type (Condo/House)

## How to Run

```bash
pip install -r requirements.txt
python train_model.py       # retrain and save the model
streamlit run app.py        # launch the prediction app
```

The app auto-detects the model's feature names, builds one input per feature, and predicts in real time.

## Tech Stack

Python, pandas, scikit-learn, matplotlib, Streamlit

## Disclaimer

For demonstration and learning purposes — not intended for real-world financial decisions.
