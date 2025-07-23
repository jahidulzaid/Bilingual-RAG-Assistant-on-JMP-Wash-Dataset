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

## ⚙️ Setup Instructions
1️⃣ Clone the Repo
```
git clone https://github.com/your-username/Bilingual-RAG-Assistant-on-JMP-Wash-Dataset.git
cd Bilingual-RAG-Assistant-on-JMP-Wash-Dataset

2️⃣ Install Dependencies
```
pip install -r requirements.txt

3️⃣ Set API Key
Create a .env file in the root:
```
GOOGLE_API_KEY=your_gemini_api_key_here

4️⃣ Run the App
```
streamlit run app.py


## 📁 Dataset
Add your cleaned JMP Wash documents in a /data/ directory.
The documents will be embedded and stored in the vector database (ChromaDB) automatically.

You said:
ImportError: cannot import name 'InvalidSchemaError' from 'pydantic._internal._generate_schema' (C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_generate_schema.py)

File "H:\Bilingual-RAG-Assistant-on-JMP-Wash-Dataset\app.py", line 7, in <module>
    from langchain_google_genai import GoogleGenerativeAI
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\langchain_google_genai\__init__.py", line 68, in <module>
    from langchain_google_genai.google_vector_store import (
    ...<2 lines>...
    )
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\langchain_google_genai\google_vector_store.py", line 55, in <module>
    class _SemanticRetriever(BaseModel):
    ...<117 lines>...
            return True
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_model_construction.py", line 226, in __new__
        k: v.info for k, v in cls.__pydantic_decorators__.computed_fields.items()
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<6 lines>...
        set_model_mocks(cls)
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_model_construction.py", line 658, in complete_model_class
    (msg := computed_field_info.deprecation_message) is not None
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\main.py", line 702, in __get_pydantic_core_schema__
    code='validate-by-alias-and-name-false',
    ^^^^^^^^^^^^^^
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_schema_generation_shared.py", line 84, in __call__
    if self._ref_mode == 'to-def':
             ^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_generate_schema.py", line 612, in generate_schema
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_generate_schema.py", line 881, in _generate_schema_inner
    # to allow existing code to call super().__get_pydantic_core_schema__ in Pydantic
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_generate_schema.py", line 693, in _model_schema
    Returns:
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_generate_schema.py", line 1073, in _generate_md_field_schema
    return self._dict_schema(Any, Any)
               ^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_generate_schema.py", line 1265, in _common_field_schema
            raise TypeError(f'alias_generator {alias_generator} must return str, not {alias.__class__}')
                 ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    
    # if priority is not set, we set to 1
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # which supports the case where the alias_generator from a child class is used
    ~~~~~
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_generate_schema.py", line 2062, in _apply_annotations
    arguments_list.append(arg_schema)
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_schema_generation_shared.py", line 84, in __call__
    if self._ref_mode == 'to-def':
             ^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_generate_schema.py", line 2043, in inner_handler
    var_kwargs_schema: core_schema.CoreSchema | None = None
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_generate_schema.py", line 886, in _generate_schema_inner
    ):
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_generate_schema.py", line 984, in match_type
    if typing_objects.is_self(obj):
               ^^^^^^^^^^^^^^^^^^^^
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_generate_schema.py", line 1744, in _dataclass_schema
    raise ValueError('Variable tuples can only have one type')
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\dataclasses.py", line 14, in <module>
    from ._internal import _dataclasses as _pydantic_dataclasses
File "C:\Users\Jahidul\anaconda3\Lib\site-packages\pydantic\_internal\_dataclasses.py", line 24, in <module>
    from ._generate_schema import GenerateSchema, InvalidSchemaError


ChatGPT said:
The error you're encountering:

pgsql
Copy
Edit
ImportError: cannot import name 'InvalidSchemaError' from 'pydantic._internal._generate_schema'
...means that the package langchain_google_genai is trying to import InvalidSchemaError from a specific part of Pydantic (pydantic._internal._generate_schema), but it's not found there. This usually indicates a version mismatch — the version of Pydantic you're using has removed or renamed this internal item.

🔍 Root Cause:
Pydantic 2.x introduced breaking changes compared to Pydantic 1.x. Internal modules like _generate_schema were refactored, and classes like InvalidSchemaError may have been removed or moved.

If langchain_google_genai is not yet compatible with Pydantic 2.x, trying to use it with Pydantic 2 will fail as you're seeing.

✅ Solution Options
Option 1: Downgrade Pydantic to 1.x
If langchain_google_genai requires Pydantic 1.x, downgrade:

bash
Copy
Edit
pip install "pydantic<2"
Note: You may need to uninstall the current version first:

bash
Copy
Edit
pip uninstall pydantic
pip install "pydantic<2"
Option 2: Use Compatible Versions of Dependencies
Run the following to check which version of langchain_google_genai you're using:

bash
Copy
Edit
pip show langchain-google-genai
Then check its documentation or PyPI page for compatible Pydantic versions.

Option 3: Upgrade langchain_google_genai (if available)
If a newer version supports Pydantic 2.x, upgrade it:

bash
Copy
Edit
pip install --upgrade langchain-google-genai
But this only works if they’ve added compatibility for Pydantic 2.

🔒 Best Practice
Pin your dependencies using a requirements.txt or pyproject.toml to avoid such version mismatches in the future.

Would you like help checking the current versions installed in your environment or generating a requirements.txt to lock them?








You said:
yes


ChatGPT said:
Great! Here's how you can check the current versions and generate a requirements.txt file:

✅ Step 1: Check Installed Package Versions
Run this in your terminal (or Anaconda Prompt):

bash
Copy
Edit
pip list
This will show all installed packages with their versions. Look for:

pydantic

langchain

langchain-google-genai

any other related packages (openai, chromadb, etc.)

✅ Step 2: Freeze Current Environment
To lock your current environment into a requirements.txt file:

bash
Copy
Edit
pip freeze > requirements.txt
This file will include exact versions like:

ini
Copy
Edit
pydantic==2.6.4
langchain==0.1.13
langchain-google-genai==0.0.9
You can then restore this exact setup later on another machine using:

bash
Copy
Edit
pip install -r requirements.txt
✅ Step 3: Fix Version Incompatibility (If Needed)
If you confirm pydantic==2.x, but langchain-google-genai expects Pydantic 1.x, do this:

Edit requirements.txt manually:

txt
Copy
Edit
pydantic<2
Then run:

bash
Copy
Edit
pip uninstall pydantic
pip install -r requirements.txt --force-reinstall
📁 Optional: Save to GitHub
If you're uploading this project to GitHub, include the requirements.txt so others can install dependencies easily.

Would you like me to review or generate a clean requirements.txt file for your LangChain project? If yes, just share your current pip list or tell me what you're using (e.g., LangChain, Google GenAI, ChromaDB, etc.).








You said:
requirements.txt
Document
which are required for this project?


ChatGPT said:
Thanks for sharing your requirements.txt. For your LangChain RAG project using Gemini (Google Generative AI), you don’t need most of these packages. I’ll list the minimal and relevant packages your project likely needs, and explain why.

✅ Essential Dependencies for Your Project
Based on your use case (Bilingual-RAG-Assistant-on-JMP-Wash-Dataset using LangChain + Google Generative AI), here’s a curated list:

txt
Copy
Edit
# Core LLM & RAG
langchain==0.3.26
langchain-community==0.3.27
langchain-core==0.3.71
langchain-google-genai==2.1.8

# Vector store
chromadb==1.0.15
sentence-transformers==5.0.0
faiss-cpu==1.11.0.post1  # Optional, if using FAISS

# Tokenization and Transformers
tiktoken==0.9.0
transformers==4.53.2
sentencepiece  # Needed for some tokenizers

# Google Generative AI support
google-ai-generativelanguage==0.6.18
google-api-core==2.25.1
google-auth==2.40.3
googleapis-common-protos==1.70.0

# Environment config
python-dotenv  # To load GOOGLE_API_KEY from .env

# Evaluation and JSON handling
pydantic==1.10.13  # ⚠️ Use 1.x, not 2.x (important)
pydantic-settings==2.1.0  # Optional
dataclasses-json==0.6.7

# Optional, for preprocessing/datasets
nltk
datasets==4.0.0

# FastAPI + Uvicorn (if using web interface)
fastapi==0.110.0
uvicorn==0.35.0
⚠️ Important Notes
You currently have pydantic==2.11.7, but langchain-google-genai (as of 2.1.8) expects Pydantic 1.x.

✅ Fix:

bash
Copy
Edit
pip uninstall pydantic
pip install "pydantic<2"
You may also remove unnecessary packages with:

bash
Copy
Edit
pip uninstall <package-name>
🛠️ Final requirements.txt (Minimal, Clean)
If you're showcasing the project on GitHub, use this:

txt
Copy
Edit
langchain==0.3.26
langchain-community==0.3.27
langchain-core==0.3.71
langchain-google-genai==2.1.8

chromadb==1.0.15
sentence-transformers==5.0.0
faiss-cpu==1.11.0.post1

tiktoken==0.9.0
transformers==4.53.2
sentencepiece

google-ai-generativelanguage==0.6.18
google-api-core==2.25.1
google-auth==2.40.3
googleapis-common-protos==1.70.0

pydantic==1.10.13
pydantic-settings==2.1.0
dataclasses-json==0.6.7

nltk
datasets==4.0.0
python-dotenv

fastapi==0.110.0
uvicorn==0.35.0
Would you like me to clean your uploaded requirements.txt and give you a downloadable version?








You said:
now write a final readmd file 


ChatGPT said:
Here is a final README.md for your project Bilingual RAG Assistant on JMP Wash Dataset — designed for GitHub and recruiters. It explains the goal, setup, usage, and architecture clearly.

markdown
Copy
Edit
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
🧪 Tech Stack
LangChain (langchain, langchain-google-genai, langchain-community)

Google Gemini Pro (google-ai-generativelanguage)

Vector Store: ChromaDB

Embedding Model: sentence-transformers

Tokenization: tiktoken, transformers

Environment Config: python-dotenv

FastAPI (optional for web interface)

⚙️ Setup Instructions
1️⃣ Clone the Repo
bash
Copy
Edit
git clone https://github.com/your-username/Bilingual-RAG-Assistant-on-JMP-Wash-Dataset.git
cd Bilingual-RAG-Assistant-on-JMP-Wash-Dataset
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Set API Key
Create a .env file in the root:

env
Copy
Edit
GOOGLE_API_KEY=your_gemini_api_key_here
4️⃣ Run the App
bash
Copy
Edit
python app.py
📁 Dataset
Add your cleaned JMP Wash documents in a /data/ directory.

The documents will be embedded and stored in the vector database (ChromaDB) automatically.

## 📊 Evaluation
You can use RAGAS or your own metrics to evaluate the assistant’s performance:

Faithfulness: Does the LLM rely on retrieved context?
Answer Relevance: Is the response complete and coherent?
Multilingual Accuracy: Are answers accurate in Bangla and English?

## 👤 Author
Jahidul Islam
Final Year, CSE
Green University of Bangladesh
Email: jahidul.cse.gub@gmail.com
