# coding_project
# Arabic Programming Learning Platform

<div align="center">

An interactive educational platform for teaching programming in Arabic using AI

</div>

## ✨ Features

* **Full Arabic Interface**: Content, lessons, and tutorials in Arabic
* **Multi-language Support**: Python, JavaScript, Java, C++, C#, Go, Rust, PHP
* **AI-Powered Content**: Dynamic content generation using Google Gemini AI
* **Interactive Quizzes**: Multiple choice questions after each lesson
* **Coding Challenges**: Practical exercises every 4 lessons
* **Smart Assistant**: Q&A system with contextual guidance
* **Progress Tracking**: Monitor learning progress throughout the curriculum

## 📦 Requirements

* Python 3.8+
* Google Gemini API key
* FastAPI
* Uvicorn (for running the server)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the project
git clone <your-repo-url>
cd arabic-programming-platform

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install fastapi uvicorn langchain-google-genai python-dotenv
```

### 2. Environment Setup

```bash
# Create environment file
echo "GOOGLE_API_KEY=your_google_gemini_api_key_here" > .env
```

**Getting the API Key:**
1. Go to Google MakerSuite
2. Create a new API key
3. Add it to the `.env` file

### 3. Run the Application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Access the application:**
* Main interface: http://localhost:8000
* API documentation: http://localhost:8000/docs
* Alternative docs: http://localhost:8000/redoc

## 📖 API Usage

### 1. Start Learning Course

```http
POST /select-language
{
  "language": "Python"
}
```

### 2. Get Lessons

```http
GET /lesson/1
```

### 3. Generate Quizzes

```http
GET /generate-quiz/1
```

### 4. Submit Answers

```http
POST /submit-quiz
{
  "lesson_id": 1,
  "answers": [0, 2, 1, 0, 3]
}
```

### 5. Ask Questions

```http
POST /ask-tutor
{
  "question": "What are functions in Python?"
}
```

### 6. Coding Challenges

```http
GET /generate-coding-challenge
POST /submit-code
{
  "code": "print('Hello World')",
  "challenge_id": "challenge_4_123456789"
}
```

## 🗂️ Curriculum Structure

1. Computer basics and operating systems
2. Python installation and environment setup
3. Learn language fundamentals
4. Variables and data types
5. Program flow control
6. Functions
7. Collections (Lists, Tuples, Sets, Dictionaries)
8. File handling
9. Using modules and libraries
10. Error handling
11. Building small practical projects
12. Package management tools (pip, venv)
13. Working on open source projects
14. Specialized libraries (NumPy, Pandas, Matplotlib)

## 🔧 Configuration

The application uses in-memory storage by default. For production:
* Add database (SQLite, PostgreSQL)
* Implement user authentication system
* Add session management
* Implement progress backup system

## 📚 Dependencies

```bash
pip install fastapi
pip install uvicorn
pip install langchain-google-genai
pip install python-dotenv
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

