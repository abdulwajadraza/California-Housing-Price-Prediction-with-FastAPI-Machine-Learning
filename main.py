# Making a Fast API app to run the California Housing Price Prediction model
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import uvicorn
import os

# Initialize the FastAPI app
app = FastAPI(
    title="California Housing Price Prediction API", 
    description="An API to predict California housing prices using a pre-trained model",
    version="1.0.0"
)

# Load the pre-trained model
try:
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Model & Scaler loaded successfully.")

except Exception as e:
    print(f"Error loading model or scaler: {e}")
    print("Ensure that 'linear_regression_model.pkl' and 'scaler.pkl' are in the current directory.")


# Define the input data model
class HouseFeatures(BaseModel):
    """ input features for the model
     MedInc: Median Income in block group
     HouseAge: Median House Age in block group
     AveRooms: Average number of rooms per household
     AveBedrms: Average number of bedrooms per household
     Population: Block group population
     AveOccup: Average number of household members
     Latitude: Block group latitude
     Longitude: Block group longitude
     """
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Define the output data model
class PredictionResponse(BaseModel):
    """ output response model """
    predicted_price: float
    message: str
    input_features: dict

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🏡 California Housing Price Prediction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f4f6f9;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                color: #333;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                max-width: 650px;
                width: 100%;
                text-align: center;
            }
            h1 {
                color: #007BFF;
                margin-bottom: 10px;
            }
            p {
                font-size: 16px;
                margin-bottom: 25px;
            }
            form {
                display: grid;
                gap: 15px;
                margin-bottom: 20px;
            }
            label {
                text-align: left;
                font-weight: bold;
                display: block;
                margin-bottom: 5px;
            }
            input {
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 6px;
                width: 100%;
            }
            button {
                background-color: #007BFF;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                transition: 0.3s;
            }
            button:hover {
                background-color: #0056b3;
            }
            .btn-secondary {
                background-color: #6c757d;
                margin-top: 10px;
            }
            .btn-secondary:hover {
                background-color: #5a6268;
            }
            .link {
                margin-top: 20px;
                display: block;
                text-decoration: none;
                color: #007BFF;
                font-weight: bold;
            }
            .result {
                margin-top: 20px;
                font-size: 18px;
                font-weight: bold;
                color: green;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏡 California Housing Price Prediction</h1>
            <p>Enter the house features below to predict its price.</p>

            <form id="predict-form">
                <div>
                    <label>Median Income (MedInc):</label>
                    <input type="number" step="any" name="MedInc" required>
                </div>
                <div>
                    <label>House Age:</label>
                    <input type="number" step="any" name="HouseAge" required>
                </div>
                <div>
                    <label>Average Rooms (AveRooms):</label>
                    <input type="number" step="any" name="AveRooms" required>
                </div>
                <div>
                    <label>Average Bedrooms (AveBedrms):</label>
                    <input type="number" step="any" name="AveBedrms" required>
                </div>
                <div>
                    <label>Population:</label>
                    <input type="number" step="any" name="Population" required>
                </div>
                <div>
                    <label>Average Occupancy (AveOccup):</label>
                    <input type="number" step="any" name="AveOccup" required>
                </div>
                <div>
                    <label>Latitude:</label>
                    <input type="number" step="any" name="Latitude" required>
                </div>
                <div>
                    <label>Longitude:</label>
                    <input type="number" step="any" name="Longitude" required>
                </div>

                <button type="submit">Predict Price</button>
            </form>

            <div class="result" id="result"></div>

            <button class="btn-secondary" 
                onclick="document.getElementById('predict-form').reset(); document.getElementById('result').innerText='';">
                Predict Another House
            </button>

            <a href="/docs" class="link">🔗 Open API Docs (Swagger)</a>
        </div>

        <script>
            document.getElementById("predict-form").addEventListener("submit", async function(event) {
                event.preventDefault();
                
                let formData = new FormData(event.target);
                let data = {};
                formData.forEach((value, key) => { data[key] = parseFloat(value); });

                let response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data)
                });

                let result = await response.json();
                document.getElementById("result").innerText = "Predicted Price: $" + result.predicted_price;
            });
        </script>
    </body>
    </html>
    """




@app.post("/predict", response_model=PredictionResponse)
async def predicted_house_price(features: HouseFeatures):
    """ Predict the housing price based on input features """
    try:
        # Convert input data to array
        input_data = np.array([[
            features.MedInc,
            features.HouseAge,
            features.AveRooms,
            features.AveBedrms,
            features.Population,
            features.AveOccup,
            features.Latitude,
            features.Longitude
        ]]) 

        input_df = pd.DataFrame(input_data)
        
        # Scale the input features
        scaled_features = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]

        # Convert prediction to readable format and multiply by 100000 to get actual price
        predicted_price = float(prediction * 100000) # Convert to actual price
        #print(f"Predicted Price: {predicted_price}")

        # Prepare response
        response = PredictionResponse(
            predicted_price=round(predicted_price, 2),
            message="Prediction successful",
            input_features=features.dict()
        )
        return response
    
    except Exception as e:
        return {"error": str(e), "message": "Prediction failed"}
    
if __name__ == "__main__":
    import uvicorn
    # Run the app with uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
