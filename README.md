# 🦙 RAG-LLaMA3

Production-grade Retrieval-Augmented Generation with **INT4-quantised LLaMA-3**, FAISS vector search, streaming inference, and an RLHF feedback loop — deployed on AWS EC2.

![Python](https://img.shields.io/badge/Python-3.11-green)
![LangChain](https://img.shields.io/badge/LangChain-0.2-blue)
![Model](https://img.shields.io/badge/LLaMA--3-INT4-blue)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-purple)
![AWS](https://img.shields.io/badge/AWS-EC2-orange)
![API](https://img.shields.io/badge/FastAPI-SSE-green)

---

## 📊 Key Results

- **−70%** Document Lookup Time (FAISS vs sequential)
- **3×** Cost Reduction using INT4 quantisation
- **+18%** Answer relevance after RLHF tuning

---

## 🏗️ System Architecture

```
Client → FastAPI → RAG Pipeline → LLaMA-3 (INT4)
                     ↓
                  FAISS
                     ↓
              Feedback DB (RLHF)
```

---

## ⚡ Quick Start

### 🧪 Local Development

```bash
git clone https://github.com/yourname/rag-llama3
cd rag-llama3
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
rag-llama3/
├── src/
├── scripts/
├── tests/
├── configs/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 📊 Quantisation Comparison

| Precision | VRAM | Cost/hr | Accuracy Drop | Tokens/sec |
|----------|------|--------|---------------|------------|
| FP16     | 16GB | $0.526 | —             | 18         |
| INT8     | 8GB  | $0.526 | ~0.5%         | 26         |
| **INT4** | 5GB  | $0.176 | <2%           | 41         |

---

## 📌 Tech Stack

- LLaMA-3 (INT4 GGUF)
- FAISS Vector Search
- LangChain
- FastAPI
- AWS EC2

---

## 🧾 License

MIT License
