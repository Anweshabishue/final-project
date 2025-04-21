from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import json  
from groq import Groq
from pypdf import PdfReader
import os
import requests 
from flask_session import Session
import re
from flask_cors import CORS
from todo_processor import TodoProcessor
from pymongo import MongoClient
# from bson.objectid import ObjectId
from datetime import datetime, timedelta, timezone
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from flask import send_from_directory
import google.generativeai as genai
from urllib.parse import quote_plus
# Load environment variables
load_dotenv()

# MongoDB Atlas Connection String
username = quote_plus("anweshabishue")
password = quote_plus("bishue@09")
MONGO_URI = f"mongodb+srv://{username}:{password}@final.6fe5roa.mongodb.net/?retryWrites=true&w=majority&appName=final"

# Define todo_collection as None initially
todo_collection = None
mongo_client = None
try:
    # Create a MongoDB client
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Test the connection
    mongo_client.admin.command('ping')
    print("Connected successfully to MongoDB Atlas!")
    
    # Initialize your databases
    todo_db = mongo_client['todo_db']
    todo_collection = todo_db['todos']
    
except Exception as e:
    print(f"Failed to connect to MongoDB Atlas: {e}")
    # You might want to exit the app or provide fallback to local MongoDB
    # For now, we'll continue but will likely fail later if MongoDB is needed

app = Flask(__name__)
bcrypt = Bcrypt(app)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'sec@2004'

# Set up Flask-SQLAlchemy for user management
db = SQLAlchemy(app)

# Set up upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# API Keys
GROQ_API_KEY = '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'
GROQ_API_KEYS = os.getenv("GROQ_API_KEY", "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")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize Google AI if API key is available
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    print("Warning: GOOGLE_API_KEY not found in environment variables")

# Initialize TodoProcessor with MongoDB collection
processor = TodoProcessor(groq_api_key=GROQ_API_KEYS, collection=todo_collection)

# Initialize TodoProcessor only if MongoDB connection was successful
if todo_collection is not None:
    processor = TodoProcessor(groq_api_key=GROQ_API_KEYS, collection=todo_collection)
else:
    print("Warning: TodoProcessor not initialized due to MongoDB connection failure")
    processor = None

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)

with app.app_context():
    db.create_all()

# Custom JSON encoder to handle MongoDB ObjectId and datetime
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Helper function to serialize MongoDB objects
def serialize_mongo(obj):
    return json.loads(json.dumps(obj, cls=MongoJSONEncoder))

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            flash("All fields are required!", "danger")
            return redirect(url_for('signup'))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered. Please log in.", "danger")
            return redirect(url_for('signup'))
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit() 

        flash("Signup successful! Please log in.", "success")
        return redirect(url_for('login'))
 
    return render_template('sign.html')
    
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id 
            flash("Login successful!", "success")
            return redirect(url_for('f3'))
        else:
            flash("Invalid email or password", "danger")
    return render_template('login.html')

@app.route('/f3')
def f3():
    if 'user_id' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))

    return render_template('f3.html')

@app.route("/")
def home():
    return render_template('f1.html')

class GrammarChecker:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def check_grammar(self, text: str) -> dict:
        prompt = f"""Please analyze the following text for grammar and spelling errors. 
        Provide corrections and explanations.
        
        Text: {text}
        
        Respond ONLY in pure JSON format, without any explanation:
        {{
            "original": "the original text",
            "corrected": "the corrected text",
            "corrections": [
                {{
                    "error": "description of the error",
                    "correction": "how it should be written",
                    "explanation": "why this correction is needed"
                }}
            ],
            "suggestions": [
                "suggestion for improving the writing"
            ]
        }}"""

        try:
            response = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                model="llama3-8b-8192",
                temperature=0.1
            )
            print("API Response:", response)
            if not response or not hasattr(response, "choices") or not response.choices:
                return {"error": "Invalid or empty response from API"}

            # Extract content safely
            content = response.choices[0].message.content if response.choices else ""
            if not content:
                return {"error": "API returned empty content"}
            print("Extracted Content:", content)

            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)

            return {"error": "Failed to extract valid JSON from response"}
        
        except json.JSONDecodeError:
            return {"error": "Failed to parse API response. Invalid JSON format."}   
        except Exception as e:
            return {
                "error": f"An error occurred: {str(e)}"
            }

@app.route('/check_grammar', methods=['POST'])
def check_grammar():
    text = request.form.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    checker = GrammarChecker()
    result = checker.check_grammar(text)

    return jsonify(result)

@app.route("/gchecker")
def gchecker():
    return render_template('check-grammar.html')

@app.route('/users')
def users():
    all_users = User.query.all()  
    return render_template('users.html', users=all_users)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def chunk_text(text, chunk_size=8000):
    """Split text into smaller chunks to handle API limits."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        if current_size + len(word) + 1 <= chunk_size:
            current_chunk.append(word)
            current_size += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def summarize_text(text):
    """Generate a summary using Groq API."""
    prompt = f"""Please provide a concise summary of the following text, highlighting the main points:

Text: {text}

Summary:"""

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
        temperature=0.3,
        max_tokens=1000
    )
    
    return chat_completion.choices[0].message.content

@app.route("/upload", methods=["GET", "POST"])
def upload_pdf():
    """Handle PDF uploads and summarization."""
    if request.method == "POST":
        if "pdf_file" not in request.files:
            return "No file uploaded", 400

        file = request.files["pdf_file"]
        if file.filename == "":
            return "No selected file", 400

        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(pdf_path)

        extracted_text = extract_text_from_pdf(pdf_path)
        if not extracted_text:
            return "Could not extract text from the PDF", 400

        chunks = chunk_text(extracted_text)
        summaries = [summarize_text(chunk) for chunk in chunks]
        final_summary = "\n\n".join(summaries)

        summary_pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], "summary_output.pdf")
        c = canvas.Canvas(summary_pdf_path)
        text_obj = c.beginText(40, 800)
        text_obj.setFont("Helvetica", 12)

        for line in final_summary.split("\n"):
            text_obj.textLine(line)
        c.drawText(text_obj)
        c.save()

        return render_template("summary_result.html", summary=final_summary)

    return render_template("upload.html")

@app.route("/download_summary")
def download_summary():
    return send_from_directory(app.config['UPLOAD_FOLDER'], "summary_output.pdf", as_attachment=True)

BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def format_response(text):
    """Formats AI response to ensure only main topics have bullet points."""
    lines = text.split("\n")
    formatted_lines = []
    
    for line in lines:
        if re.match(r"^\s*(For Loop|While Loop|Example|Do-While Loop):", line):
            formatted_lines.append(f"• {line.strip()}")
        else:
            formatted_lines.append(line.strip())
    
    return "\n".join(formatted_lines)

@app.route('/chat_ui')
def chat_ui():
     return render_template('chat.html')

@app.route('/set-subject', methods=['POST'])
def set_subject():
    """ Sets the subject for AI chat. """
    subject = request.json.get('subject')
    if not subject:
        return jsonify({"error": "No subject provided"}), 400
    session.clear()
    session['current_subject'] = subject
    session['conversation_history'] = [{
        "role": "system",
        "content": f"You are an expert in {subject}. Please focus responses on this subject."
    }]
    print(f"Subject set to: {subject}")

    return jsonify({"message": f"Subject set to: {subject}", "subject": subject})

@app.route('/chat', methods=['POST'])
def chat():
    """ Handles chat interactions. """
    if 'current_subject' not in session:
        return jsonify({"response": "Please set a subject first."})
    
    if "conversation_history" not in session:
        session["conversation_history"] = []

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"response": "No message provided."}), 400
    
    print(f"Current subject: {session['current_subject']}")
    conversation_history = session.get('conversation_history', [])
    conversation_history.append({"role": "user", "content": user_message})

    topic_check_payload = {
    "model": "llama3-8b-8192",
    "messages": [
        {
            "role": "system",
            "content": (
                f"You are an AI that checks if a user's message is related to '{session['current_subject']}'. "
                "If the message is relevant, a subtopic, or commonly discussed within this subject, return only 'ON_TOPIC'. "
                "Otherwise, return only 'OFF_TOPIC'. No explanations, just 'ON_TOPIC' or 'OFF_TOPIC'."
            )
        },
        {"role": "user", "content": user_message}
    ],
    "temperature": 0.0,
    "max_tokens": 5
}

    try:
        topic_check_response = requests.post(BASE_URL, headers=HEADERS, json=topic_check_payload)
        topic_check_response.raise_for_status()
        topic_status =topic_check_response.json()["choices"][0]["message"]["content"].strip().upper()
        print(f"Topic check result: {topic_status}")
        if topic_status == "OFF_TOPIC":
            return jsonify({
                "message": "Your question seems unrelated to the current topic. Do you want to change the subject?",
                "confirm_change": True
            })

        chat_payload = {
            "model": "llama3-8b-8192",
            "messages": conversation_history, 
            "temperature": 0.7,
            "max_tokens": 1024
        }

        ai_response = requests.post(BASE_URL, headers=HEADERS, json=chat_payload)
        ai_response.raise_for_status()
        ai_message = ai_response.json()["choices"][0]["message"]["content"]
        print(f"AI Response: {ai_message}")

        # Update conversation history
        conversation_history.append({"role": "assistant", "content": ai_message})
        session['conversation_history'] = conversation_history

        return jsonify({"response": ai_message, "confirm_change": False})

    except requests.exceptions.RequestException as e:
        print(f"API Error: {str(e)}")
        return jsonify({"response": f"Error: {str(e)}"}), 500

@app.route("/confirm-subject-change", methods=["POST"])
def confirm_subject_change():
    """ Confirms and updates the new topic when user wants to change. """
    new_subject = request.json.get('subject')
    if not new_subject:
        return jsonify({"error": "No subject provided"}), 400
    
    session.clear()
    session['current_subject'] = new_subject
    session['conversation_history'] = [{
        "role": "system",
        "content": f"You are an expert in {new_subject}. Keep responses focused on this subject."
    }]
    print(f"Subject changed to: {new_subject}")
    return jsonify({"subject": new_subject})

@app.route("/todo")
def todo():
    return render_template('text_to_todo.html')

# Route to process and save todo
@app.route("/generate_todo", methods=["POST"])
def generate_todo():
    if processor is None:
        return jsonify({"error": "MongoDB connection not available"}), 503
        
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        todo_id, structured_data = processor.process_and_save(data["text"])
        
        # Convert to JSON-serializable format
        serializable_data = serialize_mongo(structured_data)
        
        return jsonify({
            "message": "Todo saved successfully", 
            "todo_id": str(todo_id), 
            "data": serializable_data
        })
    except Exception as e:
        app.logger.error(f"Error in generate_todo: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Route to fetch all saved todos
@app.route("/get_todos", methods=["GET"])
def get_todos():
    if todo_collection is None:
        return jsonify({"error": "MongoDB connection not available"}), 503
        
    try:
        todos = list(todo_collection.find({}))
        
        # Convert MongoDB documents to JSON-serializable format
        serializable_todos = serialize_mongo(todos)
        
        return jsonify({"todos": serializable_todos})
    except Exception as e:
        app.logger.error(f"Error in get_todos: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Route to delete a todo
@app.route("/delete_todo/<todo_id>", methods=["DELETE"])
def delete_todo(todo_id):
    if processor is None:
        return jsonify({"error": "MongoDB connection not available"}), 503
        
    try:
        processor.delete_todo(todo_id)
        return jsonify({"message": "Todo deleted successfully"})
    except Exception as e:
        app.logger.error(f"Error in delete_todo: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Route to modify a todo
@app.route("/modify_todo/<todo_id>", methods=["PUT"])
def modify_todo(todo_id):
    if processor is None:
        return jsonify({"error": "MongoDB connection not available"}), 503
        
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        updated_todo = processor.modify_todo(todo_id, data["text"])
        
        # Convert to JSON-serializable format
        serializable_todo = serialize_mongo(updated_todo)
        
        return jsonify({
            "message": "Todo updated successfully", 
            "updated_todo": serializable_todo
        })
    except Exception as e:
        app.logger.error(f"Error in modify_todo: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_todos_by_date', methods=['POST'])
def get_todos_by_date():
    if todo_collection is None:
        return jsonify({"error": "MongoDB connection not available"}), 503
        
    data = request.get_json()
    date_str = data.get('date')  # Example: "2025-04-17"

    try:
        # Make it timezone-aware
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        next_day = date_obj + timedelta(days=1)

        print("Querying from:", date_obj.isoformat(), "to", next_day.isoformat())

        todos = list(todo_collection.find({
            "created_at": {
                "$gte": date_obj,
                "$lt": next_day
            }
        }))

        for todo in todos:
            todo["_id"] = str(todo["_id"])
            if isinstance(todo["created_at"], datetime):
                todo["created_at"] = todo["created_at"].isoformat()

        print("Found todos:", len(todos))

        return jsonify({"todos": todos})

    except Exception as e:
        print("Error in /get_todos_by_date:", str(e))
        return jsonify({"error": str(e)}), 500

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def build_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_answer(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)

    template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not in the provided context just say, "answer is not available in the context". 
    Don't make up any answers.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]

@app.route("/pdf_qa")
def pdf_qa():
    return render_template("pdf_Q&A.html")

@app.route("/upload_pdfs", methods=["POST"])
def upload_pdfs():
    files = request.files.getlist("pdfs[]")
    file_paths = []

    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        file_paths.append(file_path)

    raw_text = get_pdf_text(file_paths)
    chunks = get_text_chunks(raw_text)
    build_vector_store(chunks)

    return jsonify({"message": "PDFs uploaded and processed successfully."})

@app.route("/ask_question", methods=["POST"])
def ask_question():
    question = request.json.get("question")
    if not question:
        return jsonify({"error": "No question provided."}), 400

    answer = get_answer(question)
    return jsonify({"answer": answer})

# Gemini model setup
MODEL_NAME = "gemini-1.5-flash"  # or "gemini-pro"
if google_api_key:
    model = genai.GenerativeModel(MODEL_NAME)

@app.route("/question_generate")
def question_generate():
    return render_template("question.html")

@app.route("/generate_question_paper", methods=["POST"])
def generate_question_paper():
    try:
        data = request.json
        subject = data.get("subject")
        topics = data.get("topics")
        duration = data.get("duration")
        question_config = data.get("questions", {})
        difficulty = data.get("difficulty", {})

        if not subject or not topics or not duration:
            return jsonify({"error": "Missing required fields"}), 400

        # Safely convert values to integers with default of 0 for empty strings
        mc_count = int(float(question_config.get('mc', {}).get('count') or 0))
        mc_marks = int(float(question_config.get('mc', {}).get('marks') or 0))
        sa_count = int(float(question_config.get('sa', {}).get('count') or 0))
        sa_marks = int(float(question_config.get('sa', {}).get('marks') or 0))
        la_count = int(float(question_config.get('la', {}).get('count') or 0))
        la_marks = int(float(question_config.get('la', {}).get('marks') or 0))
        
        # Calculate totals
        mc_total = mc_count * mc_marks
        sa_total = sa_count * sa_marks
        la_total = la_count * la_marks
        total_marks = mc_total + sa_total + la_total
        prompt = f"""
        Create a structured and well-formatted question paper for the subject: "{subject}".
        Duration: {duration} minutes.

        Topics to include:
        {topics}

        Question Types:
        - Multiple Choice: {mc_count} questions ({mc_marks} marks each)
        - Short Answer: {sa_count} questions ({sa_marks} marks each)
        - Long Answer: {la_count} questions ({la_marks} marks each)

        IMPORTANT: The total marks for this paper MUST be exactly {total_marks}. Do not exceed this total.

        Difficulty Distribution:
        - Easy: {difficulty.get('easy')}%
        - Medium: {difficulty.get('medium')}%
        - Hard: {difficulty.get('hard')}%

        Instructions:
        - Format the question paper properly with sections and numbering.
        - Maintain clarity and proper marks indication.
        - Double-check that the total marks add up to exactly {total_marks}.

        Return only the question paper content. No explanation needed.
        """

        # Generate response using Gemini
        if not google_api_key:
            return jsonify({"error": "Google API key not configured"}), 500
            
        response = model.generate_content(prompt)
        return jsonify({"question_paper": response.text})

    except Exception as e:
        app.logger.error(f"Error in question paper generation: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Update MongoDBManager import - assuming this class exists in dbmongo_manager.py
from dbmongo_manager import MongoDBManager

# Initialize MongoDB manager with Atlas connection
mongo_manager = MongoDBManager(mongo_uri=MONGO_URI)

@app.route("/save_question_paper", methods=["POST"])
def save_question_paper():
    try:
        data = request.json
        paper_id = mongo_manager.save_paper(
            subject=data.get("subject"),
            syllabus=data.get("topics"),
            question_types=data.get("questions"),
            difficulty=data.get("difficulty"),
            pattern_id=None,
            content=data.get("content")
        )
        return jsonify({"message": "Saved", "paper_id": paper_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/logout")
def logout():
    session.pop("user_id", None)  # remove user from session
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please login to access this page.", "warning")
        return redirect(url_for("login"))
    return render_template("dashboard.html")  # Assuming you have this template

if __name__ == "__main__":
    app.run(debug=True, port=7000)
