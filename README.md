# HARMONY.AI - 🧠 A Mental Health AI Chatbot

## Overview
HARMONY.AI is an AI-powered mental health chatbot designed to provide empathetic responses and support to users. It leverages LangChain, ChromaDB, and the Llama-3 model to process and retrieve relevant information from PDF documents.

Demo Link - [HARMONY.AI](https://huggingface.co/spaces/alpinodeli/Mental_Health_AI_Chatbot)

## Features
- **Natural Language Processing**: Uses LangChain and Llama-3 for intelligent responses.
- **Document Retrieval**: Loads and processes PDF documents for enhanced contextual understanding.
- **Vector Database**: Utilizes ChromaDB for efficient data storage and retrieval.
- **User-Friendly Interface**: Built with Gradio for seamless interaction.
- **Privacy-Focused**: Ensures that conversations remain confidential and secure.

## Installation & Setup
### Prerequisites
Ensure you have Python installed along with the required dependencies.

### Clone the Repository
```bash
git clone https://github.com/AshmitJain10/HARMONY.AI.git
cd Mental-Health-AI-Chatbot
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Chatbot
```bash
python chatbot.py
```

## How It Works
1. **Initialize LLM**: Loads the Llama-3 model using `ChatGroq`.
2. **Create & Load Database**: Scans and processes PDF documents, storing them in ChromaDB.
3. **QA Retrieval Chain**: Uses LangChain to fetch relevant information.
4. **Gradio Interface**: Provides an interactive UI for user interaction.

## Usage
- Open the chatbot interface.
- Ask any mental health-related questions.
- Receive empathetic and well-informed responses.
- Type **"exit"** to end the conversation.

## Technologies Used
- **Python**: Core language for development.
- **LangChain**: For managing LLM interactions.
- **ChromaDB**: For vector-based document retrieval.
- **Gradio**: For creating the chatbot UI.
- **Llama-3**: Large language model for generating responses.

## Author
Developed by [Ashmit Jain](https://github.com/AshmitJain10/HARMONY.AI)

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! If you'd like to improve this project, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a Pull Request.

## Issues
If you encounter any issues, please report them in the [GitHub Issues](https://github.com/AshmitJain10/HARMONY.AI/issues) section.

## Acknowledgements
Special thanks to the open-source community for providing tools and resources that made this project possible.

