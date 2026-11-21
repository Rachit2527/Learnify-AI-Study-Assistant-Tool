
import os
import re
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import faiss

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

from pypdf import PdfReader


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
DB_PATH = BASE_DIR / "learnify.db"

load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Learnify AI",
    page_icon="🎓",
    layout="wide"
)

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

HF_MODEL = os.getenv(
    "HF_MODEL",
    "meta-llama/Llama-3.3-70B-Instruct"
).strip()

HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
).strip()


# ============================================================
# 2. HUGGING FACE CLIENT
# ============================================================

@st.cache_resource
def get_hf_client(token):
    return InferenceClient(token=token)


def check_hf():

    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is missing. Add it to your .env file."
        )


def ask_llm(
    prompt,
    system_prompt="You are a helpful AI study assistant.",
    temperature=0.4,
    max_tokens=2000
):

    check_hf()

    client = get_hf_client(HF_TOKEN)

    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("The AI returned an empty response.")

    return content.strip()


# ============================================================
# 3. HOSTED EMBEDDINGS - NO LOCAL PYTORCH
# ============================================================

def get_single_embedding(text):

    check_hf()

    client = get_hf_client(HF_TOKEN)

    response = client.feature_extraction(
        text,
        model=HF_EMBEDDING_MODEL
    )

    embedding = np.asarray(
        response,
        dtype=np.float32
    )

    # Handle [1, sequence_length, dimensions].
    if embedding.ndim == 3:
        embedding = embedding[0]

    # Handle token-level embeddings [sequence_length, dimensions].
    # Mean pooling converts token vectors into one document vector.
    if embedding.ndim == 2:
        embedding = embedding.mean(axis=0)

    if embedding.ndim != 1:
        raise ValueError(
            f"Unexpected embedding shape: {embedding.shape}"
        )

    if not np.isfinite(embedding).all():
        raise ValueError("Embedding API returned invalid values.")

    return embedding


def get_embeddings(texts):

    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        raise ValueError("No text provided for embedding.")

    # Individual requests avoid incorrect pooling across padded
    # sequences when a provider returns token-level embeddings.
    vectors = [
        get_single_embedding(text)
        for text in texts
    ]

    dimensions = {len(vector) for vector in vectors}

    if len(dimensions) != 1:
        raise ValueError(
            "The embedding API returned inconsistent dimensions."
        )

    embeddings = np.vstack(vectors).astype(np.float32)

    # Normalize for cosine similarity using FAISS inner product.
    faiss.normalize_L2(embeddings)

    return embeddings


# ============================================================
# 4. SESSION STATE
# ============================================================

defaults = {
    "rag_chunks": [],
    "rag_index": None,
    "rag_filename": None,
    "study_plan": None,
    "quiz": None,
    "quiz_result": None,
    "flashcards": [],
    "flashcard_index": 0,
    "flashcard_revealed": False,
    "tutor_answer": None,
    "generated_notes": None,
    "summary": None
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 5. DATABASE
# ============================================================

def get_connection():

    return sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )


def initialize_database():

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                subject TEXT,
                due_date TEXT,
                completed INTEGER DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                score INTEGER,
                total INTEGER,
                attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()


initialize_database()


# ============================================================
# 6. JSON PARSER
# ============================================================

def parse_json(response):

    response = response.strip()

    response = re.sub(
        r"^```(?:json)?\s*",
        "",
        response,
        flags=re.I
    )

    response = re.sub(
        r"\s*```$",
        "",
        response
    )

    try:
        return json.loads(response)

    except json.JSONDecodeError:

        for opening, closing in [("{", "}"), ("[", "]")]:

            start = response.find(opening)
            end = response.rfind(closing)

            if start != -1 and end > start:

                try:
                    return json.loads(
                        response[start:end + 1]
                    )
                except json.JSONDecodeError:
                    pass

        raise ValueError(
            "AI did not return valid JSON. Please try again."
        )


# ============================================================
# 7. GENERAL HELPERS
# ============================================================

def read_csv(filename):

    path = DATA_DIR / filename

    if not path.exists():
        return None

    return pd.read_csv(
        path,
        low_memory=False
    )


def split_text(
    text,
    chunk_size=900,
    overlap=150
):

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunks.append(
            text[start:end]
        )

        if end == len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# 8. PDF RAG IMPLEMENTATION
# ============================================================

def process_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    chunks = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        for chunk in split_text(text):

            chunks.append({
                "text": chunk,
                "page": page_number,
                "source": uploaded_file.name
            })

    if not chunks:
        raise ValueError(
            "No extractable text found in this PDF."
        )

    texts = [
        item["text"]
        for item in chunks
    ]

    embeddings = get_embeddings(texts)

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "Embedding count does not match PDF chunk count."
        )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    st.session_state.rag_chunks = chunks
    st.session_state.rag_index = index
    st.session_state.rag_filename = uploaded_file.name


def retrieve_documents(question, top_k=4):

    chunks = st.session_state.rag_chunks
    index = st.session_state.rag_index

    if not chunks or index is None:
        return []

    query_embedding = get_embeddings([question])

    if query_embedding.shape[1] != index.d:
        raise ValueError(
            "Query and document embedding dimensions do not match."
        )

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx >= 0:

            item = chunks[int(idx)].copy()
            item["score"] = float(score)

            results.append(item)

    return results


# ============================================================
# 9. RESUME CLASSIFICATION
# ============================================================

@st.cache_resource
def train_resume_model(csv_path, modified_time):

    df = pd.read_csv(csv_path)

    if not {"Resume", "Category"}.issubset(df.columns):
        raise ValueError(
            "Dataset must contain Resume and Category columns."
        )

    df = df.dropna(
        subset=["Resume", "Category"]
    )

    df["Resume"] = df["Resume"].astype(str)
    df["Category"] = df["Category"].astype(str)

    if df.empty:
        raise ValueError("Resume dataset is empty.")

    if df["Category"].nunique() < 2:
        raise ValueError(
            "At least two resume categories are required."
        )

    model = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=12000,
                ngram_range=(1, 2)
            )
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )

    ])

    model.fit(
        df["Resume"],
        df["Category"]
    )

    return model, len(df)


# ============================================================
# 10. BOOK RECOMMENDATION
# ============================================================

def recommend_books(book_title, n=5):

    books = read_csv("Books.csv")
    ratings = read_csv("Ratings.csv")

    if books is None or ratings is None:
        raise FileNotFoundError(
            "Books.csv and Ratings.csv are required in data/."
        )

    required_books = {"ISBN", "Book-Title"}
    required_ratings = {"ISBN", "Book-Rating"}

    if not required_books.issubset(books.columns):
        raise ValueError("Invalid Books.csv columns.")

    if not required_ratings.issubset(ratings.columns):
        raise ValueError("Invalid Ratings.csv columns.")

    ratings["Book-Rating"] = pd.to_numeric(
        ratings["Book-Rating"],
        errors="coerce"
    )

    ratings = ratings.dropna(
        subset=["ISBN", "Book-Rating"]
    )

    merged = ratings.merge(
        books[["ISBN", "Book-Title"]].drop_duplicates(),
        on="ISBN"
    )

    counts = merged.groupby(
        "Book-Title"
    )["Book-Rating"].count()

    popular = counts[
        counts >= 2
    ].index

    merged = merged[
        merged["Book-Title"].isin(popular)
    ]

    matrix = merged.pivot_table(
        index="Book-Title",
        columns="ISBN",
        values="Book-Rating",
        aggfunc="mean"
    ).fillna(0)

    if matrix.empty:
        raise ValueError(
            "Insufficient rating data."
        )

    if book_title not in matrix.index:

        matches = [
            title for title in matrix.index
            if book_title.lower() in title.lower()
        ]

        if not matches:
            raise ValueError("Book not found.")

        book_title = matches[0]

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    model.fit(matrix.values)

    distances, indices = model.kneighbors(
        matrix.loc[[book_title]].values,
        n_neighbors=min(n + 1, len(matrix))
    )

    recommendations = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        title = matrix.index[idx]

        if title != book_title:

            recommendations.append({
                "title": title,
                "similarity": round(
                    1 - float(distance),
                    3
                )
            })

    return book_title, recommendations


# ============================================================
# 11. RESUME AND JOB DESCRIPTION MATCHING
# ============================================================

def match_resume_jd(resume, jd):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform([
        resume,
        jd
    ])

    score = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    resume_words = set(
        re.findall(
            r"\b[a-zA-Z+#.]{2,}\b",
            resume.lower()
        )
    )

    jd_words = set(
        re.findall(
            r"\b[a-zA-Z+#.]{2,}\b",
            jd.lower()
        )
    )

    common = sorted(
        resume_words & jd_words
    )

    missing = sorted(
        jd_words - resume_words
    )

    return score, common, missing


# ============================================================
# 12. AI STUDY PLANNER
# ============================================================

def create_study_plan(
    subject,
    goal,
    days,
    hours,
    level
):

    prompt = f"""
Create a personalized study plan.

Subject: {subject}
Goal: {goal}
Duration: {days} days
Daily available hours: {hours}
Level: {level}

Return valid JSON only:

{{
 "overview": "overview",
 "daily_plan": [
  {{
   "day": 1,
   "focus": "topic",
   "tasks": ["task 1", "task 2"],
   "duration_hours": 2,
   "deliverable": "output"
  }}
 ],
 "revision_strategy": ["strategy"],
 "success_metrics": ["metric"]
}}

Create exactly {days} daily entries.
Do not exceed the available daily hours.
"""

    return parse_json(
        ask_llm(
            prompt,
            system_prompt="You are an expert academic planner.",
            temperature=0.3,
            max_tokens=3000
        )
    )


# ============================================================
# 13. AI QUIZ GENERATOR
# ============================================================

def create_quiz(topic, difficulty, number):

    prompt = f"""
Create {number} multiple-choice questions.

Topic: {topic}
Difficulty: {difficulty}

Return valid JSON only:

{{
 "questions": [
  {{
   "question": "question",
   "options": [
    "option 1",
    "option 2",
    "option 3",
    "option 4"
   ],
   "answer": "exact correct option text",
   "explanation": "explanation"
  }}
 ]
}}

Create exactly {number} questions.
Every question must have four options.
The answer must exactly match one option.
"""

    result = parse_json(
        ask_llm(
            prompt,
            system_prompt="You are an expert exam question creator.",
            temperature=0.4,
            max_tokens=3000
        )
    )

    questions = result.get("questions", [])

    if not isinstance(questions, list) or not questions:
        raise ValueError("No quiz questions were generated.")

    for question in questions:

        if not all(
            key in question
            for key in ["question", "options", "answer"]
        ):
            raise ValueError("Invalid quiz question format.")

        if question["answer"] not in question["options"]:
            raise ValueError(
                "A generated answer does not match its options."
            )

    return questions[:number]


# ============================================================
# 14. AI FLASHCARD GENERATOR
# ============================================================

def create_flashcards(topic, number):

    prompt = f"""
Create {number} useful revision flashcards about {topic}.

Return valid JSON only:

{{
 "flashcards": [
  {{
   "front": "question",
   "back": "answer"
  }}
 ]
}}

Create exactly {number} flashcards.
Keep answers clear and educational.
"""

    result = parse_json(
        ask_llm(
            prompt,
            system_prompt="You create effective revision flashcards.",
            temperature=0.4,
            max_tokens=2500
        )
    )

    cards = result.get("flashcards", [])

    if not isinstance(cards, list) or not cards:
        raise ValueError("No flashcards were generated.")

    return cards[:number]


# ============================================================
# 15. SIDEBAR
# ============================================================

st.sidebar.title("🎓 Learnify AI")
st.sidebar.caption("Your intelligent learning workspace")

pages = [

    "🏠 Dashboard",
    "📚 PDF RAG Chatbot",
    "🗓️ Study Planner",
    "🧠 Quiz Generator",
    "🃏 Flashcards",
    "👨‍🏫 Concept Tutor",
    "📝 AI Notes",
    "📄 Summarizer",

    "📖 Book Recommender",
    "🎯 Resume Classifier",
    "📌 Resume-JD Matcher",
    "🔍 Text Similarity",

    "✅ Task Manager",
    "📊 Learning Analytics"

]

page = st.sidebar.radio(
    "Navigation",
    pages
)

st.sidebar.divider()

st.sidebar.write("**LLM Provider:** Hugging Face")
st.sidebar.caption(f"Chat model: {HF_MODEL}")
st.sidebar.caption(f"Embedding model: {HF_EMBEDDING_MODEL}")

if HF_TOKEN:
    st.sidebar.success("API token loaded")
else:
    st.sidebar.warning("API token missing")


# ============================================================
# 16. DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🎓 Learnify AI")

    st.subheader(
        "AI-powered personalized learning platform"
    )

    st.write("""
    Learnify AI combines machine learning and generative AI
    to provide personalized learning, document-based question
    answering, automated assessments and career preparation.
    """)

    with get_connection() as conn:

        total_tasks, completed_tasks = conn.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(completed), 0)
            FROM tasks
            """
        ).fetchone()

        quiz_count = conn.execute(
            "SELECT COUNT(*) FROM quiz_attempts"
        ).fetchone()[0]

    c1, c2, c3 = st.columns(3)

    c1.metric("Total study tasks", total_tasks)
    c2.metric("Completed tasks", completed_tasks)
    c3.metric("Quiz attempts", quiz_count)

    st.markdown("### Features")

    features = [

        ("📚 RAG Chatbot", "Ask questions from uploaded PDFs."),
        ("🗓️ Study Planner", "Create personalized study roadmaps."),
        ("🧠 Quiz Generator", "Generate AI-based assessments."),
        ("🃏 Flashcards", "Create revision flashcards."),
        ("👨‍🏫 Concept Tutor", "Learn concepts interactively."),
        ("🎯 Resume AI", "Resume classification and matching.")

    ]

    cols = st.columns(3)

    for i, (title, description) in enumerate(features):

        with cols[i % 3]:

            st.info(
                f"**{title}**\n\n{description}"
            )


# ============================================================
# 17. PDF RAG CHATBOT
# ============================================================

elif page == "📚 PDF RAG Chatbot":

    st.title("📚 Chat with your PDF")

    st.write(
        "Upload a PDF and ask questions grounded in its content."
    )

    pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if pdf and st.button("Process PDF", type="primary"):

        check_hf()

        with st.spinner("Creating embeddings and FAISS index..."):

            try:

                process_pdf(pdf)

                st.success(
                    f"Processed {len(st.session_state.rag_chunks)} chunks."
                )

            except Exception as e:

                st.error(f"PDF processing failed: {e}")

    if st.session_state.rag_chunks:

        st.success(
            f"Active document: {st.session_state.rag_filename}"
        )

        question = st.text_input(
            "Ask a question"
        )

        if st.button("Ask AI", type="primary"):

            if not question.strip():

                st.warning("Enter a question.")

            else:

                with st.spinner(
                    "Retrieving context and generating answer..."
                ):

                    try:

                        retrieved = retrieve_documents(question)

                        if not retrieved:
                            raise ValueError(
                                "No relevant document chunks were found."
                            )

                        context = "\n\n".join(

                            f"""
Source: {item['source']}
Page: {item['page']}
Content: {item['text']}
"""

                            for item in retrieved
                        )

                        prompt = f"""
Answer using only the following document context.

If the answer is unavailable, say so.
Do not invent facts or references.
Mention relevant page numbers when possible.

CONTEXT:
{context}

QUESTION:
{question}
"""

                        answer = ask_llm(
                            prompt,
                            system_prompt=(
                                "You are a document-grounded RAG assistant."
                            ),
                            temperature=0.2
                        )

                        st.markdown("### Answer")
                        st.write(answer)

                        st.markdown("### Retrieved sources")

                        for item in retrieved:

                            with st.expander(
                                f"Page {item['page']} | Similarity {item['score']:.3f}"
                            ):

                                st.write(item["text"])

                    except Exception as e:

                        st.error(f"RAG query failed: {e}")


# ============================================================
# 18. STUDY PLANNER
# ============================================================

elif page == "🗓️ Study Planner":

    st.title("🗓️ AI Study Planner")

    with st.form("planner_form"):

        subject = st.text_input("Subject")

        goal = st.text_area("Learning goal")

        c1, c2, c3 = st.columns(3)

        days = c1.number_input(
            "Days",
            min_value=1,
            max_value=60,
            value=7
        )

        hours = c2.number_input(
            "Hours per day",
            min_value=0.5,
            max_value=12.0,
            value=2.0
        )

        level = c3.selectbox(
            "Level",
            ["Beginner", "Intermediate", "Advanced"]
        )

        submit = st.form_submit_button(
            "Generate Study Plan",
            type="primary"
        )

    if submit:

        if not subject.strip() or not goal.strip():

            st.warning("Fill in the subject and goal.")

        else:

            with st.spinner("Generating study roadmap..."):

                try:

                    st.session_state.study_plan = create_study_plan(
                        subject,
                        goal,
                        int(days),
                        float(hours),
                        level
                    )

                except Exception as e:

                    st.error(f"Study plan generation failed: {e}")

    plan = st.session_state.study_plan

    if plan:

        st.markdown("### Your roadmap")

        st.write(plan.get("overview", ""))

        for day in plan.get("daily_plan", []):

            with st.expander(
                f"Day {day.get('day')}: {day.get('focus')}"
            ):

                st.write(
                    f"Duration: {day.get('duration_hours')} hours"
                )

                for i, task in enumerate(day.get("tasks", [])):

                    st.checkbox(
                        task,
                        key=f"plan_{day.get('day')}_{i}"
                    )

                st.write(
                    f"Deliverable: {day.get('deliverable')}"
                )

        st.markdown("### Revision strategy")

        for item in plan.get("revision_strategy", []):
            st.write(f"- {item}")

        st.markdown("### Success metrics")

        for item in plan.get("success_metrics", []):
            st.write(f"- {item}")

        st.download_button(
            "Download plan",
            json.dumps(plan, indent=2),
            file_name="study_plan.json",
            mime="application/json"
        )


# ============================================================
# 19. QUIZ GENERATOR
# ============================================================

elif page == "🧠 Quiz Generator":

    st.title("🧠 AI Quiz Generator")

    with st.form("quiz_form"):

        topic = st.text_input("Topic")

        difficulty = st.selectbox(
            "Difficulty",
            ["Beginner", "Intermediate", "Advanced"]
        )

        number = st.slider(
            "Number of questions",
            3,
            15,
            5
        )

        generate = st.form_submit_button(
            "Generate Quiz",
            type="primary"
        )

    if generate:

        if not topic.strip():

            st.warning("Enter a topic.")

        else:

            with st.spinner("Creating quiz..."):

                try:

                    questions = create_quiz(
                        topic,
                        difficulty,
                        number
                    )

                    st.session_state.quiz = {
                        "topic": topic,
                        "questions": questions
                    }

                    st.session_state.quiz_result = None

                except Exception as e:

                    st.error(f"Quiz generation failed: {e}")

    quiz = st.session_state.quiz

    if quiz:

        st.markdown(
            f"### Quiz: {quiz['topic']}"
        )

        with st.form("answer_form"):

            answers = {}

            for i, question in enumerate(
                quiz["questions"]
            ):

                st.markdown(
                    f"**Q{i+1}. {question['question']}**"
                )

                answers[i] = st.radio(
                    "Select answer",
                    question["options"],
                    index=None,
                    key=f"answer_{i}",
                    label_visibility="collapsed"
                )

            submit = st.form_submit_button(
                "Submit Quiz",
                type="primary"
            )

        if submit:

            score = sum(
                answers.get(i) == q["answer"]
                for i, q in enumerate(quiz["questions"])
            )

            total = len(quiz["questions"])

            st.session_state.quiz_result = {
                "score": score,
                "total": total,
                "answers": answers
            }

            with get_connection() as conn:

                conn.execute(
                    """
                    INSERT INTO quiz_attempts
                    (topic, score, total)
                    VALUES (?, ?, ?)
                    """,
                    (quiz["topic"], score, total)
                )

                conn.commit()

        result = st.session_state.quiz_result

        if result:

            st.metric(
                "Your score",
                f"{result['score']} / {result['total']}"
            )

            st.progress(
                result["score"] / result["total"]
            )

            for i, q in enumerate(quiz["questions"]):

                correct = q["answer"]
                selected = result["answers"].get(i)

                if selected == correct:

                    st.success(f"Q{i+1}: Correct")

                else:

                    st.error(
                        f"Q{i+1}: Correct answer: {correct}"
                    )

                st.caption(
                    q.get("explanation", "")
                )


# ============================================================
# 20. FLASHCARDS
# ============================================================

elif page == "🃏 Flashcards":

    st.title("🃏 AI Flashcard Generator")

    with st.form("flashcard_form"):

        topic = st.text_input("Topic")

        number = st.slider(
            "Number of cards",
            3,
            20,
            8
        )

        submit = st.form_submit_button(
            "Generate Flashcards",
            type="primary"
        )

    if submit:

        if not topic.strip():

            st.warning("Enter a topic.")

        else:

            with st.spinner("Generating flashcards..."):

                try:

                    st.session_state.flashcards = create_flashcards(
                        topic,
                        number
                    )

                    st.session_state.flashcard_index = 0
                    st.session_state.flashcard_revealed = False

                except Exception as e:

                    st.error(f"Flashcard generation failed: {e}")

    cards = st.session_state.flashcards

    if cards:

        idx = st.session_state.flashcard_index
        card = cards[idx]

        st.caption(
            f"Card {idx+1} of {len(cards)}"
        )

        st.markdown(
            f"## {card['front']}"
        )

        if st.button("Reveal answer"):

            st.session_state.flashcard_revealed = True

        if st.session_state.flashcard_revealed:

            st.success(card["back"])

        c1, c2 = st.columns(2)

        with c1:

            if st.button(
                "Previous",
                disabled=idx == 0
            ):

                st.session_state.flashcard_index -= 1
                st.session_state.flashcard_revealed = False
                st.rerun()

        with c2:

            if st.button(
                "Next",
                disabled=idx == len(cards) - 1
            ):

                st.session_state.flashcard_index += 1
                st.session_state.flashcard_revealed = False
                st.rerun()

        st.download_button(
            "Download flashcards",
            pd.DataFrame(cards).to_csv(index=False),
            file_name="flashcards.csv",
            mime="text/csv"
        )


# ============================================================
# 21. CONCEPT TUTOR
# ============================================================

elif page == "👨‍🏫 Concept Tutor":

    st.title("👨‍🏫 AI Concept Tutor")

    topic = st.text_input(
        "What do you want to learn?"
    )

    level = st.selectbox(
        "Learning level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Interview preparation"
        ]
    )

    teaching_style = st.selectbox(
        "Teaching style",
        [
            "Simple explanation",
            "Step by step",
            "Real-world analogy",
            "Interview answer",
            "Socratic tutoring"
        ]
    )

    if st.button("Teach me", type="primary"):

        if not topic.strip():

            st.warning("Enter a topic.")

        else:

            with st.spinner("Preparing lesson..."):

                try:

                    answer = ask_llm(
                        f"""
Teach me {topic}.

Level: {level}
Style: {teaching_style}

Include:
1. Definition
2. Explanation
3. Practical example
4. Common mistakes
5. Interview questions
6. Quick recap
""",
                        temperature=0.5
                    )

                    st.session_state.tutor_answer = answer

                except Exception as e:

                    st.error(f"Tutor request failed: {e}")

    if st.session_state.tutor_answer:

        st.markdown(
            st.session_state.tutor_answer
        )


# ============================================================
# 22. AI NOTES GENERATOR
# ============================================================

elif page == "📝 AI Notes":

    st.title("📝 AI Notes Generator")

    topic = st.text_input(
        "Topic for notes"
    )

    level = st.selectbox(
        "Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Interview preparation"
        ]
    )

    style = st.selectbox(
        "Format",
        [
            "Structured notes",
            "Exam revision notes",
            "Interview cheat sheet",
            "Detailed chapter"
        ]
    )

    if st.button("Generate Notes", type="primary"):

        if not topic.strip():

            st.warning("Enter a topic.")

        else:

            with st.spinner("Generating notes..."):

                try:

                    notes = ask_llm(
                        f"""
Create {style} about {topic}.

Target level: {level}

Include headings, definitions, examples,
important concepts, practical applications
and a final recap.
""",
                        temperature=0.35,
                        max_tokens=3000
                    )

                    st.session_state.generated_notes = notes

                except Exception as e:

                    st.error(f"Notes generation failed: {e}")

    if st.session_state.generated_notes:

        st.markdown(
            st.session_state.generated_notes
        )

        st.download_button(
            "Download notes",
            st.session_state.generated_notes,
            file_name="learnify_notes.md",
            mime="text/markdown"
        )


# ============================================================
# 23. SUMMARIZER
# ============================================================

elif page == "📄 Summarizer":

    st.title("📄 AI Text / PDF Summarizer")

    mode = st.radio(
        "Input type",
        ["Paste text", "Upload PDF"],
        horizontal=True
    )

    content = ""

    if mode == "Paste text":

        content = st.text_area(
            "Paste text",
            height=250
        )

    else:

        pdf = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key="summary_pdf"
        )

        if pdf:

            reader = PdfReader(pdf)

            content = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

    style = st.selectbox(
        "Summary format",
        [
            "Concise",
            "Detailed",
            "Bullet points",
            "Executive summary"
        ]
    )

    if st.button("Summarize", type="primary"):

        if not content.strip():

            st.warning("Provide content.")

        else:

            with st.spinner("Summarizing..."):

                try:

                    summary = ask_llm(
                        f"""
Create a {style} summary of this content.

Preserve important information.
Do not add facts not present in the source.

CONTENT:
{content[:45000]}
""",
                        temperature=0.25,
                        max_tokens=2500
                    )

                    st.session_state.summary = summary

                except Exception as e:

                    st.error(f"Summarization failed: {e}")

    if st.session_state.summary:

        st.markdown(
            st.session_state.summary
        )

        st.download_button(
            "Download summary",
            st.session_state.summary,
            file_name="summary.md",
            mime="text/markdown"
        )


# ============================================================
# 24. BOOK RECOMMENDER
# ============================================================

elif page == "📖 Book Recommender":

    st.title("📖 ML Book Recommendation System")

    books = read_csv("Books.csv")

    if books is None:

        st.info(
            "Add Books.csv and Ratings.csv to the data folder."
        )

    elif not {"Book-Title", "ISBN"}.issubset(books.columns):

        st.error("Books.csv is missing required columns.")

    else:

        titles = sorted(
            books["Book-Title"]
            .dropna()
            .astype(str)
            .unique()
        )

        selected = st.selectbox(
            "Select a book",
            titles
        )

        n = st.slider(
            "Number of recommendations",
            3,
            10,
            5
        )

        if st.button("Recommend", type="primary"):

            try:

                source, results = recommend_books(
                    selected,
                    n
                )

                st.success(
                    f"Books similar to {source}"
                )

                if not results:
                    st.info("No recommendations found.")

                for item in results:

                    st.write(
                        f"**{item['title']}** "
                        f"— Similarity: {item['similarity']}"
                    )

            except Exception as e:

                st.error(f"Recommendation failed: {e}")


# ============================================================
# 25. RESUME CLASSIFIER
# ============================================================

elif page == "🎯 Resume Classifier":

    st.title("🎯 Resume Category Prediction")

    st.write(
        "TF-IDF + Random Forest classification."
    )

    resume = st.text_area(
        "Paste resume text",
        height=250
    )

    resume_file = st.file_uploader(
        "Or upload a .txt resume",
        type=["txt"]
    )

    if st.button("Predict Category", type="primary"):

        if resume_file:

            resume = resume_file.getvalue().decode(
                "utf-8",
                errors="ignore"
            )

        csv_path = DATA_DIR / "resume_dataset.csv"

        if not resume.strip():

            st.warning("Provide resume content.")

        elif not csv_path.exists():

            st.error(
                "Add data/resume_dataset.csv with Resume and Category columns."
            )

        else:

            try:

                model, rows = train_resume_model(
                    str(csv_path),
                    csv_path.stat().st_mtime
                )

                prediction = model.predict([resume])[0]

                st.success(
                    f"Predicted category: {prediction}"
                )

                st.caption(
                    f"Training records: {rows}"
                )

                probabilities = model.predict_proba([resume])[0]

                classes = model.classes_

                ranked = sorted(
                    zip(classes, probabilities),
                    key=lambda x: x[1],
                    reverse=True
                )

                st.markdown("### Top category scores")

                for label, score in ranked[:5]:

                    st.write(
                        f"{label}: {score:.2%}"
                    )

            except Exception as e:

                st.error(f"Classification failed: {e}")


# ============================================================
# 26. RESUME-JD MATCHER
# ============================================================

elif page == "📌 Resume-JD Matcher":

    st.title("📌 Resume–Job Description Matcher")

    resume = st.text_area(
        "Resume",
        height=220
    )

    jd = st.text_area(
        "Job description",
        height=220
    )

    if st.button("Analyze Match", type="primary"):

        if not resume.strip() or not jd.strip():

            st.warning("Enter both texts.")

        else:

            try:

                score, common, missing = match_resume_jd(
                    resume,
                    jd
                )

                st.metric(
                    "TF-IDF similarity",
                    f"{score:.1%}"
                )

                st.progress(
                    float(score)
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.markdown("### Matching keywords")

                    st.write(
                        ", ".join(common[:30])
                    )

                with c2:

                    st.markdown("### Potential missing keywords")

                    st.write(
                        ", ".join(missing[:30])
                    )

                st.caption(
                    "This is a lexical similarity score, not a hiring probability."
                )

            except Exception as e:

                st.error(f"Matching failed: {e}")


# ============================================================
# 27. TEXT SIMILARITY
# ============================================================

elif page == "🔍 Text Similarity":

    st.title("🔍 Text Similarity Analyzer")

    text_a = st.text_area(
        "Text A",
        height=180
    )

    text_b = st.text_area(
        "Text B",
        height=180
    )

    if st.button("Calculate Similarity", type="primary"):

        if not text_a.strip() or not text_b.strip():

            st.warning("Enter both texts.")

        else:

            try:

                vectorizer = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2)
                )

                matrix = vectorizer.fit_transform([
                    text_a,
                    text_b
                ])

                score = cosine_similarity(
                    matrix[0:1],
                    matrix[1:2]
                )[0][0]

                st.metric(
                    "Similarity score",
                    f"{score:.2%}"
                )

                st.progress(float(score))

            except Exception as e:

                st.error(f"Similarity calculation failed: {e}")


# ============================================================
# 28. TASK MANAGER
# ============================================================

elif page == "✅ Task Manager":

    st.title("✅ Study Task Manager")

    with st.form("task_form", clear_on_submit=True):

        title = st.text_input("Task title")

        subject = st.text_input("Subject")

        due_date = st.date_input("Due date")

        submit = st.form_submit_button(
            "Add Task",
            type="primary"
        )

    if submit:

        if title.strip():

            with get_connection() as conn:

                conn.execute(
                    """
                    INSERT INTO tasks
                    (title, subject, due_date)
                    VALUES (?, ?, ?)
                    """,
                    (title, subject, str(due_date))
                )

                conn.commit()

            st.success("Task added.")

        else:

            st.warning("Enter a task title.")

    with get_connection() as conn:

        tasks = pd.read_sql_query(
            "SELECT * FROM tasks ORDER BY completed, due_date",
            conn
        )

    if tasks.empty:

        st.info("No tasks yet.")

    else:

        for _, task in tasks.iterrows():

            c1, c2, c3 = st.columns([0.1, 0.7, 0.2])

            done = c1.checkbox(
                "Done",
                value=bool(task["completed"]),
                key=f"task_{task['id']}"
            )

            c2.write(
                f"**{task['title']}**"
            )

            c2.caption(
                f"{task['subject']} | Due: {task['due_date']}"
            )

            if done != bool(task["completed"]):

                with get_connection() as conn:

                    conn.execute(
                        "UPDATE tasks SET completed=? WHERE id=?",
                        (int(done), int(task["id"]))
                    )

                    conn.commit()

                st.rerun()

            if c3.button(
                "Delete",
                key=f"delete_{task['id']}"
            ):

                with get_connection() as conn:

                    conn.execute(
                        "DELETE FROM tasks WHERE id=?",
                        (int(task["id"]),)
                    )

                    conn.commit()

                st.rerun()


# ============================================================
# 29. LEARNING ANALYTICS
# ============================================================

elif page == "📊 Learning Analytics":

    st.title("📊 Learning Analytics Dashboard")

    with get_connection() as conn:

        tasks = pd.read_sql_query(
            "SELECT * FROM tasks",
            conn
        )

        quizzes = pd.read_sql_query(
            "SELECT * FROM quiz_attempts ORDER BY attempted_at",
            conn
        )

    total = len(tasks)

    completed = (
        int(tasks["completed"].sum())
        if total else 0
    )

    quiz_count = len(quizzes)

    average_score = 0

    if quiz_count:

        average_score = (
            quizzes["score"] / quizzes["total"]
        ).mean() * 100

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Completed tasks",
        f"{completed}/{total}"
    )

    c2.metric(
        "Quiz attempts",
        quiz_count
    )

    c3.metric(
        "Average quiz score",
        f"{average_score:.1f}%"
    )

    if not tasks.empty:

        st.markdown("### Task completion")

        task_counts = tasks["completed"].map({
            0: "Pending",
            1: "Completed"
        }).value_counts()

        st.bar_chart(task_counts)

    if not quizzes.empty:

        st.markdown("### Quiz performance")

        quizzes["percentage"] = (
            quizzes["score"] / quizzes["total"] * 100
        )

        st.line_chart(
            quizzes.set_index("attempted_at")["percentage"]
        )

        st.dataframe(
            quizzes[
                [
                    "topic",
                    "score",
                    "total",
                    "percentage",
                    "attempted_at"
                ]
            ],
            use_container_width=True
        )

    if tasks.empty and quizzes.empty:

        st.info(
            "Complete study tasks and attempt quizzes to see analytics."
        )


# ============================================================
# 30. FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Learnify AI | Hybrid ML + GenAI Learning Platform"
)

st.sidebar.caption(
    "Built with Streamlit, Hugging Face, FAISS and Scikit-learn"
)
