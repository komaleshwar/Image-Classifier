# AI Image Classifier

This repository contains code and resources for an AI image classifier. The project is built using a CNN model to classify images as either AI-generated or real.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

The image classifier project aims to distinguish between AI-generated images and real images using a pre-trained Convolutional Neural Network. The project utilizes the Streamlit framework for creating an interactive web application.

## Installation

To install the necessary dependencies, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the dependencies listed in the `Requirements.txt` file:

<pre> pip install -r Requirements.txt </pre>


## Usage

To run the image classifier application, use the following command:

<pre> streamlit run app.py </pre>


This will start the Streamlit application, and you can access it in your web browser.

Once the application is running, follow these steps:

1. Click on the "Browse Files" button to select an image file (in JPG or PNG or JPEG or WEBM format).
2. Wait for the model to load and process the image.
3. The application will display the uploaded image along with the prediction result.
4. The prediction will indicate whether the image is most likely AI-generated or a real image.
