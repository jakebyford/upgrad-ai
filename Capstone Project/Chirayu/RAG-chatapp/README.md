# Chat with Model

## Overview

This project is a Streamlit application that allows users to interact with a language model to generate responses based on input questions and optional context. The application uses the Hugging Face `transformers` library to load a pre-trained model and tokenizer, and it leverages Streamlit for the user interface.

## Features

- **User Interface**: A Streamlit-based web interface for user interaction.
- **Model Integration**: Uses a pre-trained language model for generating responses.
- **Contextual Input**: Allows users to provide context along with their questions for more accurate responses.
- **Tokenization**: Utilizes a tokenizer to format the input prompt.
- **Response Generation**: Generates responses using the same model that formats the input.

## Prerequisites

- Python 3.7 or later
- Streamlit
- Hugging Face `transformers` library
- Requests library (for API integration, if needed)
- Dotenv library (for environment variable management)

## Installation

1. **Create a Virtual Environment**:
    ```python
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
3. **Set Up Environment Variables**:
    - Create a `.env` file in the root directory of the project.
    - Add the following environment variables to the `.env` file:
        ```sh
        ACCESS_TOKEN=<your_hugging_face_access_token>
        ```

## USAGE

1. **Run the Streamlit App**:
    ```sh
    streamlit run app.py
    ```
2. **Interact with the App**:
    - Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
    - Enter your question in the "Enter your question" input field.
    - Optionally, provide context in the "Enter context (optional)" text area.
    - Set the maximum number of tokens for the generated response using the "Max new tokens" slider.
    - Click the "Generate Response" button to get the generated response from the model.

## NOTE
The code given for the model runs locally and the model and the embedding model i.e `sentence-transformer` gets stored in the `models` folder.
The sample files were the files that were used for the demo.

## OUTPUT

![WorkingImage](/images/Working_img.png)

The Terminal images for them were

![Terminal1](/images/Terminal_output_1.png)
![Terminal2](/images/Terminal_output_2.png)

The generated tensors look like

![Tensors1](/images/Tensor_output_1.png)
![Tensors2](/images/Tensor_output_2.png)
![Tensors3](/images/Tensor_output_Final.png)

The video and image for another prompt

<video src="images/video.mp4" width="1280" height="720" controls></video>

![Image](/images/Working_img_2.png)