# AI Matters Virtual Assistant

This project is a virtual assistant for AI Matters, providing a web-based interface to answer queries about AI Matters using a FastAPI backend, a vector database for document retrieval, and a frontend built with HTML, CSS, and JavaScript. The assistant leverages a language model (LLM), a vector store for context-aware responses, and integrates image generation capabilities using Hugging Face's Stable Diffusion model via Gradio Client. Features include autocomplete suggestions, related questions, source linking, and image generation for visual queries.

## Project Structure

- **Backend**:
  - `scraper.py`: Scrapes content from `ai-matters.eu` and saves it to `data/ai_matters_data.jsonl`.
  - `vector_db_creation.ipynb`: Jupyter Notebook that processes the scraped data and creates a Chroma vector database.
  - `main.py`: FastAPI application serving the virtual assistant API, including endpoints for query handling, autocomplete suggestions, related questions, and image generation.
  - `trie_utils.py`: Implements a Trie-based autocomplete system for efficient suggestion retrieval.
  - `image_gen.py`: Handles image generation using the Gradio Client to interact with Hugging Face's Stable Diffusion model.
  - `suggestions_db.py`: Contains enhanced suggestion databases for autocomplete and related questions.
- **Frontend**:
  - `index.html`: Main HTML file for the user interface.
  - `styles.css`: Styles for the frontend interface, including support for displaying generated images.
  - `script.js`: JavaScript for frontend interactivity, including fetching answers, rendering suggestions, handling chat flow, and displaying generated images.

## Prerequisites

- **Python**: Version 3.8 or higher.
- **uv**: A Python project manager for managing virtual environments and dependencies (for backend and frontend).
- **Git**: For cloning the repository.
- **Internet Connection**: Required for fetching external resources, running the API, and accessing Hugging Face's image generation service.
- **Hugging Face API Token**: Required for image generation via Gradio Client.
- **Google Colab with T4 GPU**: Required to run `vector_db_creation.ipynb` for creating the vector database.

## Setup Instructions (Using uv and Google Colab)

Follow these steps to set up and run the project:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Mustafa-Ahmed-Rizwan/AI-Matters.git
   cd your-project
   ```

2. **Install uv (Optional)**
   If `uv` is not already installed, install it using pip:

   ```bash
   pip install uv
   ```

3. **Create and Activate a Virtual Environment**
   Create a virtual environment using `uv` and activate it:

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install Dependencies**
   Sync the dependencies from the `uv.lock` file:

   ```bash
   uv sync
   ```

5. **Set Up Environment Variables**
   Create a `.env` file in the project root and add your Groq API key and Hugging Face API token:

   ```bash
   echo "GROQ_API_KEY=your-groq-api-key" > .env
   echo "HF_API_TOKEN=your-huggingface-api-token" >> .env
   ```

   Replace `your-groq-api-key` with your actual Groq API key from [https://console.groq.com/](https://console.groq.com/).
   Replace `your-huggingface-api-token` with your Hugging Face API token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). When creating the HF token, ensure you check all permissions under the **User Permissions** section, including **Repositories** and **Inference** options, to allow access to the required Hugging Face Spaces and models.

6. **Create the Vector Database**
   To create the Chroma vector database, use Google Colab with a T4 GPU:

   - Open [Google Colab](https://colab.research.google.com/).
   - Upload `vector_db_creation.ipynb` to your Colab environment.
   - Connect to a T4 GPU runtime: In Colab, go to **Runtime** > **Change runtime type** > Select **T4 GPU** under Hardware accelerator.
   - Ensure the `data/ai_matters_data.jsonl` file (generated from `scraper.py`) is uploaded to your Colab environment or accessible via a mounted Google Drive.
   - Run each cell in the `vector_db_creation.ipynb` notebook sequentially to process the scraped data and create the vector database.
   - After running, download the generated vector database files (from the `vector_db3` directory) to your local project directory for use by the backend.

7. **Run the Application**
   The project consists of a frontend and a backend, which need to be run in separate terminals.

   **Backend**:
   Open a terminal, navigate to the project root, and start the FastAPI server:

   ```bash
   uv run main.py
   ```

   The backend will run on `http://localhost:8000`.

   **Frontend**:
   Open another terminal, navigate to the frontend directory (if applicable, or stay in the project root if `index.html` is there), and start a simple HTTP server:

   ```bash
   cd frontend  # If your frontend files are in a 'frontend' directory; skip if in root
   python -m http.server 8080
   ```

   The frontend will be accessible at `http://localhost:8080`.

8. **Access the Application**
   Open your browser and navigate to `http://localhost:8080` to use the AI Matters Virtual Assistant.

## Usage

- **Search**: Enter a query in the search bar to get answers about AI Matters. Use the arrow keys to navigate autocomplete suggestions, and press Enter to select a suggestion or submit a query.
- **Quick Help**: Click on suggested questions in the "Quick Help" section to get instant answers.
- **Related Questions**: After receiving an answer, explore related questions to dive deeper into the topic.
- **Image Generation**: Enter queries like "generate an image of [description]" to create images using Stable Diffusion. A stop button is available to cancel image generation if needed.
- **Sources**: Answers include links to relevant sources from `ai-matters.eu` when available.
- **Feedback**: Use the thumbs-up or thumbs-down buttons to provide feedback on answer quality.

## Notes

- The `uv.lock` file ensures reproducible dependency installations. Ensure it is committed to the repository.
- The `data/ai_matters_data.jsonl` file is generated by running `scraper.py`. If you need to re-scrape data, run:
  ```bash
   python scraper.py
  ```
  Then, recreate the vector database by running all cells in `vector_db_creation.ipynb` in Google Colab with a T4 GPU.
- The backend requires a valid Groq API key and Hugging Face API token in the `.env` file.
- Image generation relies on the Hugging Face Space (default: `stabilityai/stable-diffusion`). Ensure the model space is accessible and the API token has the necessary permissions.
- The frontend uses a simple HTTP server for development. For production, consider a more robust server setup.

## Troubleshooting

- **Dependency Issues**: Ensure `uv sync` completes successfully. If errors occur, check the `uv.lock` file or regenerate it with `uv lock`.
- **API Errors**: Verify the Groq API key and Hugging Face API token are correct, with appropriate permissions for Repositories and Inference, and that the server is running at `http://localhost:8000`.
- **Frontend Not Loading**: Ensure you are running the HTTP server in the correct directory containing `index.html`.
- **Vector Database Issues**: If the vector store fails to load, re-run all cells in `vector_db_creation.ipynb` in Google Colab with a T4 GPU, ensuring `data/ai_matters_data.jsonl` is accessible.
- **Image Generation Issues**: Check the Hugging Face API token permissions and ensure the specified model space (e.g., `stabilityai/stable-diffusion`) is available. Verify network connectivity for external API calls.

## Contributing

Contributions are welcome! Please fork the repository, create a branch, and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
