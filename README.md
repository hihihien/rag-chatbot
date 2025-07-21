## Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/hihihien/rag-chatbot.git
cd rag-chatbot
```

---

### 2. Setup Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate     
pip install -r requirements.txt

# Start backend server
python serve_api.py

### 🔹 3. Setup Frontend

```bash
cd ../frontend
npm install
npm run dev
```
> Runs at `http://localhost:3000`
