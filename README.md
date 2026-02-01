# Nyay Buddy 

**Nyay Buddy** is an intelligent Legal Named Entity Recognition (NER) and analysis tool designed to assist legal professionals, researchers, and common citizens in understanding complex legal documents. Leveraging a fine-tuned BERT model, it automatically extracts and highlights key legal entities and provides an AI-powered chat assistant to answer questions about the document.

##  Features

-   **High-Precision NER:** Automatically identifies and categorizes legal entities such as **Case Numbers, Court Names, Judges, Dates, Petitioners, Respondents, Statutes, and more** using a fine-tuned `Legal-BERT` model.
-   **Smart Entity Highlighting:** Color-coded highlighting of extracted entities within the document text for quick scanning and review.
-   **Interactive AI Assistant:** Built-in chat interface powered by Groq (Llama 3) to answer questions, summarize cases, or explain legal terms found in the document.
-   **Visual Analytics:** Dynamic charts and graphs powered by Plotly to visualize entity distribution and model confidence scores.
-   **Multi-Format Support:** Seamlessly handles both **PDF** and **TXT** file uploads.
-   **Robust Fallback:** Function in offline demo mode if the deep learning model or API is unavailable.

##  Tech Stack

### Frontend & Interface
-   **Framework:** Streamlit (Python)
-   **Visualization:** Plotly Express
-   **Styling:** Custom CSS with Inter font and glassmorphism elements

### Backend & AI Core
-   **Language:** Python 3.8+
-   **NLP Model:** Hugging Face Transformers (BertForTokenClassification)
-   **Base Model:** Legal-BERT (Fine-tuned)
-   **LLM Integration:** Groq API (Llama 3)
-   **Data Processing:** Pandas, NumPy, PyPDF

##  Prerequisites

Before you begin, ensure you have the following installed:
-   **Python** (v3.8 or higher)
-   **pip** (Python Package Installer)
-   **Git**

##  Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Prajwal1516/nyay-buddy.git
    cd nyay-buddy
    ```

2.  **Environment Setup**
    It is recommended to use a virtual environment.
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Install the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration**
    -   Create a `.streamlit/secrets.toml` file or a `.env` file to store your API keys.
    -   Add your Groq API Key:
        ```toml
        # .streamlit/secrets.toml
        GROQ_API_KEY = "your_groq_api_key_here"
        ```

##  Running the Application

To start the Nyay Buddy dashboard, run the Streamlit application from the root directory:

```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

##  Model Information

The project is designed to use a fine-tuned `legal-bert-ner` model.
-   **Local Model:** If you have the fine-tuned model files, place them in a directory named `legal-bert-ner` in the project root.
-   **Demo Mode:** If the local model is not found, the application uses a rule-based demonstration mode so you can still explore the UI and features.

##  Usage

1.  **Upload Document:**
    -   Click "Browse files" to upload a Legal Judgment or Case File (PDF or TXT).
    
2.  **Analyze:**
    -   Click the **" Analyze Text"** button.
    -   Wait for the BERT model to extract entities.

3.  **Review Results:**
    -   **Highlighted Text:** Read the document with color-coded entities.
    -   **Entities Table:** View a structured list of all found entities and confidence scores.
    -   **Charts:** Analyze the frequency and confidence of different entity types.

4.  **Ask the AI:**
    -   Use the " Legal Assistant" chat at the bottom to ask questions like "Who is the petitioner?" or "Summarize the verdict".

##  Contributing

Contributions are welcome to make justice more accessible!

1.  Fork the project
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

