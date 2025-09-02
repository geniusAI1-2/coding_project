from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini AI using LangChain
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(" Please set your GOOGLE_API_KEY! Get it from: https://makersuite.google.com/app/apikey")

# Initialize ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

app = FastAPI(
    title="Arabic Programming Learning Platform", 
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
memory_storage = {
    "students": {},
    "courses": {},
    "sessions": {},
    "current_session": None,
    "student_progress": {},
    "quiz_cache": {},
    "coding_quizzes": {}  # New storage for coding quizzes
}

# Pydantic models
class LanguageSelection(BaseModel):
    language: str

class QuestionRequest(BaseModel):
    question: str

class QuizSubmission(BaseModel):
    lesson_id: int
    answers: List[int]

class CodeSubmission(BaseModel):
    code: str
    challenge_id: str

# Utility functions
def get_lesson_structure():
    """Returns the 14-lesson structure in Arabic"""
    return [
        "أساسيات الحاسوب ونظام التشغيل",
        "تثبيت بايثون وضبط بيئة العمل", 
        "تعلم أساسيات اللغة",
        "التعامل مع المتغيرات وأنواع البيانات",
        "التحكم في سير البرنامج باستخدام الشروط والتكرار",
        "الدوال",
        "المجموعات مثل Lists و Tuples و Sets و Dictionaries",
        "التعامل مع الملفات",
        "استخدام الموديولات والمكتبات",
        "التعامل مع الأخطاء",
        "بناء مشاريع عملية صغيرة",
        "استخدام أدوات إدارة الحزم مثل pip و venv",
        "العمل على مشاريع مفتوحة المصدر أو حل تمارين برمجية",
        "تعلم مكتبات متخصصة مثل NumPy و Pandas و Matplotlib لتحليل البيانات"
    ]

def clean_json_response(response_text: str) -> str:
    """Clean and extract JSON from AI response"""
    try:
        # Remove markdown code blocks if present
        cleaned = response_text.strip()
        
        # Remove markdown code block markers
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]  # Remove ```json
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]   # Remove ```
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]  # Remove ending ```
        
        cleaned = cleaned.strip()
        
        # Find JSON content between braces - look for the main object
        brace_count = 0
        start_idx = -1
        end_idx = -1
        
        for i, char in enumerate(cleaned):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    end_idx = i + 1
                    break
        
        if start_idx != -1 and end_idx != -1:
            json_str = cleaned[start_idx:end_idx]
            # Test if it's valid JSON
            json.loads(json_str)
            return json_str
        else:
            # Fallback: try to parse the whole cleaned string
            json.loads(cleaned)
            return cleaned
            
    except Exception as e:
        print(f"Error cleaning JSON response: {e}")
        print(f"Original response (first 500 chars): {response_text[:500]}...")
        raise ValueError(f"Could not extract valid JSON from response: {str(e)}")

def safe_json_loads(json_str: str, default=None):
    """Safely load JSON with error handling"""
    if not json_str or json_str.strip() == "":
        return default or {}
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Problematic JSON string: {json_str[:200]}...")
        return default or {}

async def generate_curriculum(language: str) -> dict:
    """Generate curriculum for a programming language using Gemini AI"""
    lessons_structure = get_lesson_structure()
    
    prompt = f"""
    أنت مدرس خبير في البرمجة. أريد منك إنشاء منهج تعليمي للغة البرمجة {language} باللغة العربية.
    
    يجب أن يتكون المنهج من 14 درسًا بناءً على هذا الهيكل:
    {json.dumps(lessons_structure, ensure_ascii=False, indent=2)}
    
    لكل درس، أريد:
    1. عنوان الدرس مُعدّل ليناسب لغة {language}
    2. وصف موجز للمحتوى (2-3 جمل)
    3. الأهداف التعليمية (3-4 نقاط)
    
    أرجع النتيجة في صيغة JSON صحيحة بهذا الشكل فقط، بدون أي نص إضافي أو تنسيق markdown:
    {{
        "language": "{language}",
        "lessons": [
            {{
                "lesson_number": 1,
                "title": "العنوان",
                "description": "الوصف",
                "objectives": ["هدف 1", "هدف 2", "هدف 3"]
            }}
        ]
    }}
    
    مهم جداً: لا تضع النتيجة في code blocks أو أي تنسيق markdown. أرجع JSON خام فقط.
    تأكد من أن المحتوى مناسب للمبتدئين ومرتب بشكل منطقي.
    """
    
    try:
        response = await llm.ainvoke(prompt)
        print(f"Debug: Raw AI response (first 200 chars): {response.content[:200]}...")
        
        cleaned_response = clean_json_response(response.content)
        print(f"Debug: Cleaned response (first 200 chars): {cleaned_response[:200]}...")
        
        # Test if it's valid JSON
        curriculum = json.loads(cleaned_response)
        print(f"Debug: Successfully parsed JSON with {len(curriculum.get('lessons', []))} lessons")
        
        return curriculum
    except Exception as e:
        print(f"Error generating curriculum: {e}")
        # Return a fallback curriculum
        fallback_curriculum = {
            "language": language,
            "lessons": [
                {
                    "lesson_number": i + 1,
                    "title": title,
                    "description": f"تعلم {title} في لغة {language}",
                    "objectives": [f"فهم أساسيات {title}", f"تطبيق {title} عملياً", f"حل المشاكل باستخدام {title}"]
                }
                for i, title in enumerate(lessons_structure)
            ]
        }
        return fallback_curriculum

async def generate_lesson_content(language: str, lesson_number: int, lesson_title: str) -> dict:
    """Generate detailed lesson content using Gemini AI"""
    
    prompt = f"""
    أنت مدرس خبير في لغة البرمجة {language}. أريد منك إنشاء محتوى تفصيلي للدرس رقم {lesson_number}: "{lesson_title}" باللغة العربية.
    
    يجب أن يتضمن المحتوى:
    1. مقدمة عن الموضوع (100-150 كلمة)
    2. الشرح التفصيلي مع الأمثلة (300-500 كلمة)
    3. أمثلة برمجية عملية مع التعليقات باللغة العربية (3-5 أمثلة)
    4. نصائح مهمة للطلاب
    5. ملخص سريع للدرس
    
    أرجع النتيجة في صيغة JSON صحيحة فقط، بدون أي نص إضافي أو تنسيق markdown:
    {{
        "introduction": "المقدمة",
        "detailed_explanation": "الشرح التفصيلي",
        "code_examples": [
            {{
                "title": "عنوان المثال",
                "code": "الكود",
                "explanation": "شرح المثال"
            }}
        ],
        "tips": ["نصيحة 1", "نصيحة 2"],
        "summary": "الملخص"
    }}
    
    مهم جداً: لا تضع النتيجة في code blocks أو أي تنسيق markdown. أرجع JSON خام فقط.
    تأكد من أن الأمثلة صحيحة ومناسبة للمستوى المبتدئ.
    """
    
    try:
        response = await llm.ainvoke(prompt)
        print(f"Debug: Lesson content AI response (first 200 chars): {response.content[:200]}...")
        
        cleaned_response = clean_json_response(response.content)
        print(f"Debug: Cleaned lesson content (first 200 chars): {cleaned_response[:200]}...")
        
        content = json.loads(cleaned_response)
        print(f"Debug: Successfully parsed lesson content with keys: {content.keys()}")
        
        return content
    except Exception as e:
        print(f"Error generating lesson content: {e}")
        # Return fallback content
        return {
            "introduction": f"مرحباً بك في الدرس {lesson_number}: {lesson_title}. في هذا الدرس سنتعلم المفاهيم الأساسية والمهمة في {lesson_title}.",
            "detailed_explanation": f"في هذا الدرس سنتعلم {lesson_title} في لغة {language}. هذا الموضوع مهم جداً للمبتدئين ويشكل أساساً قوياً لفهم المفاهيم المتقدمة في البرمجة.",
            "code_examples": [
                {
                    "title": "مثال بسيط",
                    "code": f"// هذا مثال بسيط في {language}\n// {lesson_title}\nprint('مرحباً بالعالم');",
                    "explanation": "هذا المثال يوضح المفهوم الأساسي"
                }
            ],
            "tips": ["تدرب كثيراً على الأمثلة", "اقرأ الكود بعناية", "لا تتردد في طرح الأسئلة"],
            "summary": f"تعلمنا في هذا الدرس {lesson_title} وأهمية هذا المفهوم في البرمجة"
        }

async def generate_quiz_questions(language: str, lesson_number: int, lesson_title: str) -> dict:
    """Generate 5 MCQ questions for a lesson"""
    
    prompt = f"""
    أنت مدرس خبير في لغة البرمجة {language}. أريد منك إنشاء 5 أسئلة اختيار متعدد للدرس رقم {lesson_number}: "{lesson_title}" باللغة العربية.
    
    لكل سؤال:
    1. السؤال واضح ومباشر
    2. 4 خيارات (أ، ب، ج، د)
    3. إجابة واحدة صحيحة فقط
    4. تغطي النقاط المهمة في الدرس
    
    أرجع النتيجة في صيغة JSON صحيحة فقط، بدون أي نص إضافي:
    {{
        "questions": [
            {{
                "question": "نص السؤال",
                "options": ["الخيار أ", "الخيار ب", "الخيار ج", "الخيار د"],
                "correct_answer": 0,
                "explanation": "شرح الإجابة الصحيحة"
            }}
        ]
    }}
    
    تأكد من أن الأسئلة متنوعة وتختبر الفهم الحقيقي وليس الحفظ فقط.
    """
    
    try:
        response = await llm.ainvoke(prompt)
        cleaned_response = clean_json_response(response.content)
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error generating quiz questions: {e}")
        # Return fallback questions
        return {
            "questions": [
                {
                    "question": f"ما هو الموضوع الرئيسي للدرس {lesson_number}؟",
                    "options": [lesson_title, "موضوع آخر", "لا أعرف", "كل ما سبق"],
                    "correct_answer": 0,
                    "explanation": f"الإجابة الصحيحة هي {lesson_title}"
                }
            ]
        }

async def generate_coding_challenge(language: str, lessons_completed: int) -> dict:
    """Generate a coding challenge after every 4 lessons"""
    
    # Create a timestamp for the challenge ID
    timestamp = int(datetime.now().timestamp())
    
    prompt = f"""
    أنت مدرس خبير في لغة البرمجة {language}. الطالب قد أكمل {lessons_completed} دروس حتى الآن.
    
    أريد منك إنشاء تحدي برمجي مناسب لمستوى الطالب الحالي. التحدي يجب أن:
    1. يكون مناسبًا للمستوى الحالي (بعد {lessons_completed} دروس)
    2. يكون عمليًا وواقعيًا
    3. يتضمن وصفًا واضحًا للمشكلة
    4. يتضمن أمثلة للإدخال والإخراج المتوقع
    5. يكون باللغة العربية
    
    أرجع النتيجة في صيغة JSON صحيحة فقط، بدون أي نص إضافي:
    {{
        "challenge_id": "معرف فريد للتحدي",
        "title": "عنوان التحدي",
        "description": "وصف مفصل للتحدي والمشكلة التي يجب حلها",
        "requirements": ["المتطلب 1", "المتطلب 2"],
        "example_input": "مثال للإدخال",
        "example_output": "مثال للإخراج المتوقع",
        "hints": ["تلميح 1", "تلميح 2"]
    }}
    
    استخدم معرفًا فريدًا للتحدي مثل "challenge_{lessons_completed}_{timestamp}"
    """
    
    try:
        response = await llm.ainvoke(prompt)
        cleaned_response = clean_json_response(response.content)
        challenge_data = json.loads(cleaned_response)
        
        # Ensure challenge_id is unique
        challenge_id = challenge_data.get("challenge_id", f"challenge_{lessons_completed}_{timestamp}")
        challenge_data["challenge_id"] = challenge_id
        
        return challenge_data
    except Exception as e:
        print(f"Error generating coding challenge: {e}")
        # Return fallback challenge
        return {
            "challenge_id": f"challenge_{lessons_completed}_{timestamp}",
            "title": f"تحدي برمجي بعد {lessons_completed} دروس",
            "description": f"قم بكتابة برنامج يحل المشكلة التالية بناءً على ما تعلمته في الدروس {lessons_completed-3}-{lessons_completed}",
            "requirements": ["يجب أن يحل البرنامج المشكلة المطلوبة", "يجب أن يكون الكود نظيفًا وواضحًا"],
            "example_input": "المدخلات المطلوبة",
            "example_output": "المخرجات المتوقعة",
            "hints": ["فكر في استخدام الدوال", "تأكد من معالجة جميع الحالات"]
        }

async def evaluate_code(language: str, code: str, challenge_id: str, challenge_data: dict) -> dict:
    """Evaluate student's code and provide feedback"""
    
    prompt = f"""
    أنت مدرس خبير في لغة البرمجة {language}. الطالب قد كتب الكود التالي لتحدي برمجي:
    
    الكود:
    {code}
    
    معلومات التحدي:
    - العنوان: {challenge_data.get('title', '')}
    - الوصف: {challenge_data.get('description', '')}
    - المتطلبات: {json.dumps(challenge_data.get('requirements', []), ensure_ascii=False)}
    - مثال الإدخال: {challenge_data.get('example_input', '')}
    - مثال الإخراج: {challenge_data.get('example_output', '')}
    
    قم بتقييم الكود بناءً على:
    1. هل يحل المشكلة بشكل صحيح؟
    2. هل يلبي جميع المتطلبات؟
    3. هل هناك أخطاء في الكود؟
    4. جودة الكود ووضوحه
    
    إذا كان هناك أخطاء، قدم تلميحات لمساعدة الطالب على تصحيحها دون إعطاء الحل مباشرة.
    
    أرجع النتيجة في صيغة JSON صحيحة فقط، بدون أي نص إضافي:
    {{
        "is_correct": true/false,
        "score": 0-100,
        "feedback": "ملاحظات عامة على الكود",
        "errors": ["قائمة بالأخطاء إن وجدت"],
        "hints": ["تلميحات لتحسين الكود أو تصحيح الأخطاء"],
        "suggestions": ["اقتراحات لتحسين جودة الكود"]
    }}
    
    كن مشجعًا وتعليميًا في ملاحظاتك.
    """
    
    try:
        response = await llm.ainvoke(prompt)
        cleaned_response = clean_json_response(response.content)
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error evaluating code: {e}")
        # Return fallback evaluation
        return {
            "is_correct": False,
            "score": 0,
            "feedback": "عذرًا، حدث خطأ في تقييم الكود. يرجى المحاولة مرة أخرى.",
            "errors": ["لا يمكن تقييم الكود حاليًا"],
            "hints": ["تأكد من صحة بناء الجملة في الكود"],
            "suggestions": ["راجع الدروس السابقة للتعرف على المفاهيم الأساسية"]
        }

async def get_ai_tutor_response(question: str, language: str, lesson_number: int, chat_history: list) -> str:
    """Get AI tutor response while keeping student on track"""
    
    # Build context from chat history
    context = "\n".join([f"الطالب: {msg['user']}\nالمدرس: {msg['assistant']}" for msg in chat_history[-5:]])
    
    prompt = f"""
    أنت مدرس ذكي ومتخصص في تعليم لغة البرمجة {language} باللغة العربية. الطالب حاليًا في الدرس رقم {lesson_number}.
    
    السياق السابق للمحادثة:
    {context}
    
    سؤال الطالب الحالي: {question}
    
    قواعد مهمة:
    1. أجب فقط عن الأسئلة المتعلقة بالدرس الحالي أو الدروس السابقة
    2. إذا سأل عن موضوع من درس متقدم، قل له بلطف أنه سيتعلم هذا في الدروس القادمة
    3. استخدم أمثلة برمجية بسيطة ومفهومة
    4. كن صبورًا ومشجعًا
    5. إذا كان السؤال غير واضح، اطلب التوضيح
    6. إذا كان السؤال خارج البرمجة تمامًا، وجهه بلطف للعودة للموضوع
    
    أجب بطريقة ودودة وتعليمية باللغة العربية.
    """
    
    try:
        response = await llm.ainvoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error getting AI tutor response: {e}")
        return "عذراً، حدث خطأ في النظام. يرجى إعادة المحاولة."

# API Endpoints
@app.get("/health")
async def health_check():
    """Check if the API is working"""
    return {
        "status": "healthy",
        "ai_model": "gemini-1.5-flash",
        "message": "API is running successfully"
    }

@app.post("/select-language")
async def select_language(language_data: LanguageSelection):
    """Select programming language and generate course"""
    
    try:
        # Generate curriculum using AI
        curriculum = await generate_curriculum(language_data.language)
        
        # Store in memory
        session_id = f"session_{datetime.now().timestamp()}"
        memory_storage["current_session"] = {
            "session_id": session_id,
            "language": language_data.language,
            "curriculum": curriculum,
            "current_lesson": 1,
            "completed_lessons": [],
            "chat_history": [],
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "message": f"تم إنشاء كورس {language_data.language} بنجاح!",
            "session_id": session_id,
            "language": language_data.language,
            "current_lesson": 1,
            "total_lessons": len(curriculum.get("lessons", [])),
            "curriculum_overview": curriculum.get("lessons", [])[:3]  # First 3 lessons preview
        }
        
    except Exception as e:
        print(f"Error in select_language: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating course: {str(e)}")

@app.get("/lesson/{lesson_number}")
async def get_lesson(lesson_number: int):
    """Get lesson content"""
    
    try:
        current_session = memory_storage.get("current_session")
        if not current_session:
            raise HTTPException(status_code=404, detail="No active session. Please select a language first.")
        
        if lesson_number > current_session["current_lesson"]:
            raise HTTPException(status_code=403, detail="Complete previous lessons first")
        
        curriculum = current_session["curriculum"]
        if not curriculum or "lessons" not in curriculum:
            raise HTTPException(status_code=500, detail="Curriculum not found")
        
        if len(curriculum["lessons"]) < lesson_number:
            raise HTTPException(status_code=404, detail=f"Lesson {lesson_number} not found")
        
        lesson_info = curriculum["lessons"][lesson_number - 1]
        
        # Generate detailed lesson content
        content = await generate_lesson_content(
            current_session["language"], 
            lesson_number, 
            lesson_info.get("title", f"الدرس {lesson_number}")
        )
        
        return {
            "lesson_number": lesson_number,
            "title": lesson_info.get("title"),
            "description": lesson_info.get("description"),
            "objectives": lesson_info.get("objectives", []),
            "content": content,
            "language": current_session["language"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_lesson: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching lesson: {str(e)}")

@app.post("/ask-tutor")
async def ask_tutor(question_data: QuestionRequest):
    """Ask question to AI tutor"""
    
    try:
        current_session = memory_storage.get("current_session")
        if not current_session:
            raise HTTPException(status_code=404, detail="No active session. Please select a language first.")
        
        # Get AI response
        ai_response = await get_ai_tutor_response(
            question_data.question, 
            current_session["language"], 
            current_session["current_lesson"], 
            current_session["chat_history"]
        )
        
        # Update chat history
        chat_entry = {
            "user": question_data.question,
            "assistant": ai_response,
            "timestamp": datetime.now().isoformat()
        }
        current_session["chat_history"].append(chat_entry)
        
        return {
            "response": ai_response,
            "session_id": current_session["session_id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in ask_tutor: {e}")
        raise HTTPException(status_code=500, detail="Error processing question")

@app.get("/generate-quiz/{lesson_number}")
async def generate_quiz(lesson_number: int):
    """Generate quiz for a lesson"""
    
    try:
        current_session = memory_storage.get("current_session")
        if not current_session:
            raise HTTPException(status_code=404, detail="No active session. Please select a language first.")
        
        curriculum = current_session["curriculum"]
        if not curriculum or "lessons" not in curriculum:
            raise HTTPException(status_code=500, detail="Curriculum not found")
        
        if len(curriculum["lessons"]) < lesson_number:
            raise HTTPException(status_code=404, detail=f"Lesson {lesson_number} not found")
        
        lesson_info = curriculum["lessons"][lesson_number - 1]
        session_id = current_session["session_id"]
        
        # Check if quiz already exists in cache
        cache_key = f"{session_id}_lesson_{lesson_number}"
        if cache_key in memory_storage["quiz_cache"]:
            cached_quiz = memory_storage["quiz_cache"][cache_key]
            print(f"Debug: Using cached quiz for lesson {lesson_number}")
            return {
                "lesson_number": lesson_number,
                "lesson_title": lesson_info.get("title"),
                "questions": cached_quiz,
                "cached": True
            }
        
        # Generate new quiz using AI
        questions = await generate_quiz_questions(
            current_session["language"], 
            lesson_number, 
            lesson_info.get("title", f"الدرس {lesson_number}")
        )
        
        # Cache the generated quiz
        memory_storage["quiz_cache"][cache_key] = questions
        print(f"Debug: Generated and cached new quiz for lesson {lesson_number}")
        
        return {
            "lesson_number": lesson_number,
            "lesson_title": lesson_info.get("title"),
            "questions": questions,
            "cached": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in generate_quiz: {e}")
        raise HTTPException(status_code=500, detail="Error generating quiz")

@app.post("/submit-quiz")
async def submit_quiz(submission: QuizSubmission):
    """Submit and evaluate quiz answers"""
    
    try:
        current_session = memory_storage.get("current_session")
        if not current_session:
            raise HTTPException(status_code=404, detail="No active session. Please select a language first.")
        
        curriculum = current_session["curriculum"]
        lesson_number = submission.lesson_id  # Using lesson_id as lesson_number
        
        if len(curriculum["lessons"]) < lesson_number:
            raise HTTPException(status_code=404, detail=f"Lesson {lesson_number} not found")
        
        lesson_info = curriculum["lessons"][lesson_number - 1]
        session_id = current_session["session_id"]
        
        # Get quiz from cache - this ensures we use the same questions
        cache_key = f"{session_id}_lesson_{lesson_number}"
        if cache_key in memory_storage["quiz_cache"]:
            questions_data = memory_storage["quiz_cache"][cache_key]
            print(f"Debug: Using cached quiz for evaluation of lesson {lesson_number}")
        else:
            print(f"Warning: No cached quiz found for lesson {lesson_number}, generating new one")
            # Generate quiz questions to compare answers (fallback)
            questions_data = await generate_quiz_questions(
                current_session["language"], 
                lesson_number, 
                lesson_info.get("title", f"الدرس {lesson_number}")
            )
            # Cache it for future use
            memory_storage["quiz_cache"][cache_key] = questions_data
        
        questions = questions_data.get("questions", [])
        
        if not questions:
            raise HTTPException(status_code=400, detail="No questions found")
        
        # Evaluate answers
        correct_count = 0
        results = []
        
        for i, user_answer in enumerate(submission.answers):
            if i < len(questions):
                correct_answer = questions[i].get("correct_answer", 0)
                is_correct = user_answer == correct_answer
                if is_correct:
                    correct_count += 1
                
                results.append({
                    "question_number": i + 1,
                    "question_text": questions[i].get("question", ""),
                    "user_answer": user_answer,
                    "correct_answer": correct_answer,
                    "is_correct": is_correct,
                    "explanation": questions[i].get("explanation", "لا يوجد شرح متاح")
                })
        
        if not results:
            raise HTTPException(status_code=400, detail="No valid answers to evaluate")
        
        score = (correct_count / len(questions)) * 100
        passed = score >= 70  # 70% passing grade
        
        # If passed, update progress
        if passed:
            if lesson_number not in current_session["completed_lessons"]:
                current_session["completed_lessons"].append(lesson_number)
            
            # Move to next lesson if this was the current lesson
            if lesson_number == current_session["current_lesson"] and lesson_number < 14:
                current_session["current_lesson"] += 1
        
        return {
            "score": score,
            "passed": passed,
            "correct_answers": correct_count,
            "total_questions": len(questions),
            "results": results,
            "message": "تهانينا! لقد نجحت في الاختبار" if passed else "للأسف، لم تجتز الاختبار. حاول مرة أخرى",
            "next_lesson_available": current_session["current_lesson"] if passed else None,
            "quiz_was_cached": cache_key in memory_storage["quiz_cache"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in submit_quiz: {e}")
        raise HTTPException(status_code=500, detail="Error submitting quiz")

@app.get("/session-status")
async def get_session_status():
    """Get current session status and progress"""
    
    try:
        current_session = memory_storage.get("current_session")
        if not current_session:
            return {
                "has_active_session": False,
                "message": "No active session"
            }
        
        return {
            "has_active_session": True,
            "session_id": current_session["session_id"],
            "language": current_session["language"],
            "current_lesson": current_session["current_lesson"],
            "completed_lessons": current_session["completed_lessons"],
            "total_lessons": len(current_session["curriculum"].get("lessons", [])),
            "progress_percentage": (len(current_session["completed_lessons"]) * 100) / 14,
            "chat_history_count": len(current_session["chat_history"]),
            "created_at": current_session["created_at"]
        }
        
    except Exception as e:
        print(f"Error in get_session_status: {e}")
        raise HTTPException(status_code=500, detail="Error fetching session status")

@app.get("/available-languages")
async def get_available_languages():
    """Get list of supported programming languages"""
    
    return {
        "languages": [
            {"name": "Python", "description": "لغة برمجة سهلة ومناسبة للمبتدئين"},
            {"name": "JavaScript", "description": "لغة برمجة الويب والتطبيقات التفاعلية"},
            {"name": "Java", "description": "لغة برمجة قوية ومناسبة للمشاريع الكبيرة"},
            {"name": "C++", "description": "لغة برمجة سريعة ومناسبة للألعاب والأنظمة"},
            {"name": "C#", "description": "لغة برمجة من مايكروسوفت لتطوير التطبيقات"},
            {"name": "Go", "description": "لغة برمجة حديثة وسريعة من جوجل"},
            {"name": "Rust", "description": "لغة برمجة آمنة وسريعة للأنظمة"},
            {"name": "PHP", "description": "لغة برمجة مخصصة لتطوير مواقع الويب"}
        ]
    }

@app.get("/generate-coding-challenge")
async def generate_coding_challenge_endpoint():
    """Generate a coding challenge after every 4 lessons"""
    
    try:
        current_session = memory_storage.get("current_session")
        if not current_session:
            raise HTTPException(status_code=404, detail="No active session. Please select a language first.")
        
        completed_lessons = len(current_session["completed_lessons"])
        
        # Only generate challenges after every 4 lessons
        if completed_lessons < 4:
            raise HTTPException(
                status_code=400, 
                detail=f"يتم إنشاء التحديات البرمجية بعد إكمال 4 دروس على الأقل. لقد أكملت {completed_lessons} دروس فقط حتى الآن."
            )
        
        # Check if we already have a challenge for this milestone
        session_id = current_session["session_id"]
        challenge_milestone = (completed_lessons // 4) * 4
        existing_challenge_key = f"{session_id}_challenge_{challenge_milestone}"
        
        if existing_challenge_key in memory_storage["coding_quizzes"]:
            challenge = memory_storage["coding_quizzes"][existing_challenge_key]["challenge_data"]
            return {
                "challenge": challenge,
                "message": f"تم إيجاد تحدي برمجي سابق بعد إكمال {challenge_milestone} دروس",
                "is_cached": True
            }
        
        # Generate new coding challenge
        challenge = await generate_coding_challenge(
            current_session["language"], 
            challenge_milestone
        )
        
        # Store challenge in memory
        memory_storage["coding_quizzes"][existing_challenge_key] = {
            "challenge_data": challenge,
            "session_id": session_id,
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "challenge": challenge,
            "message": f"تم إنشاء تحدي برمجي جديد بعد إكمال {challenge_milestone} دروس",
            "is_cached": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in generate_coding_challenge: {e}")
        raise HTTPException(status_code=500, detail="Error generating coding challenge")

@app.post("/submit-code")
async def submit_code(submission: CodeSubmission):
    """Submit and evaluate code for a coding challenge"""
    
    try:
        current_session = memory_storage.get("current_session")
        if not current_session:
            raise HTTPException(status_code=404, detail="No active session. Please select a language first.")
        
        # Get challenge data
        challenge_id = submission.challenge_id
        if challenge_id not in memory_storage["coding_quizzes"]:
            raise HTTPException(status_code=404, detail="Challenge not found")
        
        challenge_data = memory_storage["coding_quizzes"][challenge_id]["challenge_data"]
        
        # Evaluate the code
        evaluation = await evaluate_code(
            current_session["language"],
            submission.code,
            challenge_id,
            challenge_data
        )
        
        return {
            "evaluation": evaluation,
            "challenge_id": challenge_id,
            "challenge_title": challenge_data.get("title", "")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in submit_code: {e}")
        raise HTTPException(status_code=500, detail="Error evaluating code")

if __name__ == "__main__":
    import uvicorn
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\n Received interrupt signal. Shutting down gracefully...")
        sys.exit(0)
    
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print(" Starting Arabic Programming Learning Platform...")
        print(" Make sure to set your GOOGLE_API_KEY environment variable")
        print(" API will be available at: http://localhost:8000")
        print(" API documentation at: http://localhost:8000/docs")
        print("  Press Ctrl+C to stop")
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n✅ Server stopped gracefully")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        print("👋 Goodbye!")