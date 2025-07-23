# 🌐 Bilingual RAG Assistant on JMP Wash Dataset

This project is a **Retrieval-Augmented Generation (RAG) Assistant** designed for the **Joint Monitoring Programme (JMP) Wash dataset**, capable of answering queries in both **Bangla** and **English**.

It uses **LangChain** + **Google Gemini (Generative AI)** with a **vector-based document retrieval system (ChromaDB)** to build a smart, bilingual knowledge assistant that responds with grounded answers from documents.

---

## ✨ Features

- ✅ **Bilingual Query Support**: Accepts queries in Bangla and English.
- 🔍 **Vector Search** with ChromaDB and Sentence Transformers.
- 🧠 **RAG Pipeline** using LangChain + Google Generative AI (`gemini-pro`).
- 🔐 Secure API key management using `.env`.
- 🛠️ Modular and scalable design.
- 📄 Dataset-backed responses for real-world, domain-specific applications.

---

## 🧩 Architecture

```text
[User Query (Bangla/English)]
        |
        v
[Language Detection]
        |
        v
[Vector DB (Chroma) ←→ LangChain Retriever]
        |
        v
[Prompt Template + Context → Gemini-Pro (LLM)]
        |
        v
[Final Answer]
---
## ⚙️ Setup Instructions
1️⃣ Clone the Repo
```bash
git clone https://github.com/your-username/Bilingual-RAG-Assistant-on-JMP-Wash-Dataset.git
cd Bilingual-RAG-Assistant-on-JMP-Wash-Dataset
```

2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

3️⃣ Set API Key
Create a .env file in the root:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```
4️⃣ Run the App
```bash
streamlit run app.py
```
---
## 📁 Dataset
Add your cleaned JMP Wash documents in a /data/ directory.
The documents will be embedded and stored in the vector database (ChromaDB) automatically.

---
## 📊 Evaluation
You can use RAGAS or your own metrics to evaluate the assistant’s performance:

Faithfulness: Does the LLM rely on retrieved context?
Answer Relevance: Is the response complete and coherent?
Multilingual Accuracy: Are answers accurate in Bangla and English?

---

## 👤 Author
Jahidul Islam

