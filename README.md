# Forecasting with Prophet

This Streamlit app allows users to upload a CSV file containing time-series data and generate forecasts using Facebook Prophet.

## Features

- **Upload Data**: Users can upload their own CSV file containing time-series data.
- **Preview Data**: The app provides a preview of the uploaded data.
- **Generate Forecast**: After uploading the data, the app generates forecasts for the next year using Facebook Prophet.
- **Visualize Forecast Components**: Users can visualize various components of the forecast, such as trend, seasonality, and holidays.

## Usage

1. **Upload Data**: Click on the "Upload a CSV file" button to upload your own CSV file containing time-series data.
2. **Preview Data**: Once the file is uploaded, the app displays a preview of the data.
3. **Generate Forecast**: The app automatically generates forecasts for the next year based on the uploaded data.
4. **Visualize Forecast Components**: Users can visualize different components of the forecast using the provided plots.

## Deployment

This app is deployed and accessible online at [Forecasting with Prophet](https://skill-extraction-with-bert-yvf7zfcahggh5zyi6zfgwn.streamlit.app/).

## Installation

To run this app locally, follow these steps:

1. Clone this repository:

    ```bash
    git clone https://github.com/azraimahadan/skill-extraction-with-bert.git
    ```

2. Navigate to the project directory:

    ```bash
    cd skill-extraction-with-bert
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Requirements

- Python 3.x
- Streamlit
- Prophet
- pandas
- matplotlib

## License

This project is licensed under the MIT License.
