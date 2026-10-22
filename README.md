<div align="center">

  <h1>🌾 Rainfall &amp; Crop Yield Prediction</h1>

  <p>
    <strong>A Machine Learning system for regional precipitation forecasting and agricultural yield estimation.</strong>
  </p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  </p>

</div>

<hr />

<h2>📌 Overview</h2>
<p>
  Accurate weather and yield forecasting empowers farmers and policymakers to optimize crop selection, irrigation schedules, and resource planning. This repository houses end-to-end data pipelines, exploratory analysis, trained regression models, and web deployment interfaces for:
</p>
<ul>
  <li><strong>Rainfall Prediction:</strong> Estimating regional precipitation trends from historical weather parameters.</li>
  <li><strong>Crop Yield Forecasting:</strong> Predicting crop production based on soil nutrients, rainfall, crop variety, and area cultivated.</li>
</ul>

<hr />

<h2>🗂️ Repository Structure</h2>
<pre>
├── annual_rainfall_prediction_dataset.csv   # Historical rainfall dataset
├── crop_yield_database.csv                  # Agricultural parameters and harvest yields
├── Rain_pred.ipynb                          # Rainfall EDA, feature engineering & modeling
├── Crop_Yield_pred.ipynb                    # Crop yield preprocessing & model evaluation
├── rainfall_pred.pkl                        # Serialized model for rainfall inference
├── crop_yield_model.pkl                     # Serialized model for crop yield inference
├── app.py                                   # Primary web application interface
├── app2.py                                  # Secondary deployment pipeline
├── app3.py                                  # Experimental interface
└── requerments.txt                          # Project dependencies
</pre>

<hr />

<h2>⚙️ Setup &amp; Installation</h2>

<h3>1. Clone the Repository</h3>
<pre><code>git clone https://github.com/rohitpatil261003/Rainfall_and_Crop_Yield_Prediction.git
cd Rainfall_and_Crop_Yield_Prediction</code></pre>

<h3>2. Create a Virtual Environment</h3>
<pre><code># Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python
