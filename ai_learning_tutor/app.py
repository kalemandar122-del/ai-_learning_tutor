
import os
import json
import sqlite3
from datetime import datetime

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ---------------- CONFIGURATION ----------------

load_dotenv()

DB_PATH = "learning_tutor.db"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

st.set_page_config(
    page_title="AI Personalized Learning Tutor",
    page_icon="🎓",
    layout="wide"
)


# ---------------- DATABASE ----------------

def get_db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                student_id TEXT PRIMARY KEY,
                level TEXT DEFAULT 'Beginner',
                style TEXT DEFAULT 'Simple and easy',
                goals TEXT DEFAULT ''
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                role TEXT,
                content TEXT,
                created_at TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                topic TEXT,
                score INTEGER,
                total INTEGER,
                created_at TEXT
            )
        """)


def load_profile(student_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT level, style, goals FROM profiles WHERE student_id=?",
            (student_id,)
        ).fetchone()

    if row:
        return {
            "level": row[0],
            "style": row[1],
            "goals": row[2]
        }

    return {
        "level": "Beginner",
        "style": "Simple and easy",
        "goals": ""
    }


def save_profile(student_id, level, style, goals):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO profiles(student_id, level, style, goals)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(student_id) DO UPDATE SET
                level=excluded.level,
                style=excluded.style,
                goals=excluded.goals
        """, (student_id, level, style, goals))


def save_message(student_id, role, content):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO messages(student_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            student_id,
            role,
            content,
            datetime.now().isoformat(timespec="seconds")
        ))


def load_messages(student_id, limit=20):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT role, content FROM (
                SELECT id, role, content
                FROM messages
                WHERE student_id=?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
        """, (student_id, limit)).fetchall()

    return [
        {"role": role, "content": content}
        for role, content in rows
    ]


def save_quiz_result(student_id, topic, score, total):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO quiz_results
            (student_id, topic, score, total, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            student_id,
            topic,
            score,
            total,
            datetime.now().isoformat(timespec="seconds")
        ))


init_db()


# ---------------- GROQ API CONNECTION ----------------

def get_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key or api_key == "your_new_groq_api_key_here":
        raise ValueError(
            "Groq API key missing. Add GROQ_API_KEY to your .env file."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )


def ask_ai(messages, temperature=0.4):
    client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature
    )

    return response.choices[0].message.content or ""


def get_system_prompt(level, style, goals, subject, topic):
    return f"""
You are an AI Personalized Learning Tutor.

Student level: {level}
Preferred explanation style: {style}
Learning goals: {goals or "Not specified"}
Subject: {subject}
Topic: {topic or "General"}

Instructions:
1. Explain concepts according to the student's level.
2. Use simple language and examples when appropriate.
3. Break difficult topics into steps.
4. Encourage understanding and practice.
5. Use headings and bullet points where useful.
6. If you are uncertain, say so rather than inventing facts.
"""


# ---------------- SIDEBAR ----------------

st.title("🎓 AI Personalized Learning Tutor")
st.caption("Your personal AI tutor powered by Groq")

with st.sidebar:
    st.header("Student Profile")

    student_id = st.text_input(
        "Student ID",
        value="student1"
    ).strip()

    if not student_id:
        st.warning("Enter a student ID.")
        st.stop()

    profile = load_profile(student_id)

    levels = ["Beginner", "Intermediate", "Advanced"]

    styles = [
        "Simple and easy",
        "Step-by-step",
        "Examples first",
        "Detailed and technical"
    ]

    level = st.selectbox(
        "Learning Level",
        levels,
        index=levels.index(profile["level"])
        if profile["level"] in levels else 0
    )

    style = st.selectbox(
        "Explanation Style",
        styles,
        index=styles.index(profile["style"])
        if profile["style"] in styles else 0
    )

    goals = st.text_area(
        "Learning Goals",
        value=profile["goals"],
        placeholder="Example: Learn Python and prepare for exams"
    )

    if st.button("Save Profile"):
        save_profile(student_id, level, style, goals)
        st.success("Profile saved!")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "💬 AI Tutor",
            "📝 Quiz Generator",
            "📅 Study Planner",
            "📊 Learning Progress",
            "🧠 Memory Settings"
        ]
    )


# ---------------- HOME ----------------

if page == "🏠 Home":
    st.header("Welcome to Your Learning Dashboard")

    with get_db() as conn:
        results = conn.execute("""
            SELECT score, total
            FROM quiz_results
            WHERE student_id=?
        """, (student_id,)).fetchall()

    total_questions = sum(row[1] for row in results)
    total_correct = sum(row[0] for row in results)

    accuracy = (
        total_correct / total_questions * 100
        if total_questions else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Learning Level", level)
    col2.metric("Quizzes Completed", len(results))
    col3.metric("Quiz Accuracy", f"{accuracy:.1f}%")

    st.subheader("Available Features")

    st.markdown("""
    - **AI Tutor:** Ask questions and understand difficult concepts.
    - **Quiz Generator:** Generate quizzes and check your answers.
    - **Study Planner:** Create a personalized study schedule.
    - **Learning Progress:** Track quiz performance.
    - **Conversational Memory:** Review or clear saved chat history.
    """)

    st.info(f"Your preferred learning style is: {style}")


# ---------------- AI TUTOR ----------------

elif page == "💬 AI Tutor":
    st.header("Chat with Your AI Tutor")

    subject = st.selectbox(
        "Select Subject",
        [
            "Computer Science",
            "Mathematics",
            "Science",
            "Artificial Intelligence",
            "Data Analytics",
            "Other"
        ]
    )

    topic = st.text_input(
        "Topic",
        placeholder="Example: Machine Learning"
    )

    if st.session_state.get("active_student") != student_id:
        st.session_state.active_student = student_id
        st.session_state.chat_history = load_messages(student_id)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_messages(student_id)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask your learning question...")

    if prompt:
        history = st.session_state.chat_history[-10:]

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Your tutor is thinking..."):
                try:
                    system_prompt = get_system_prompt(
                        level, style, goals, subject, topic
                    )

                    answer = ask_ai(
                        [
                            {
                                "role": "system",
                                "content": system_prompt
                            }
                        ] + history + [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    st.markdown(answer)

                    save_message(student_id, "user", prompt)
                    save_message(student_id, "assistant", answer)

                    st.session_state.chat_history = (
                        history + [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer}
                        ]
                    )

                except Exception as e:
                    st.error(f"Error connecting to Groq: {e}")
                    st.info(
                        "Check your API key, internet connection, "
                        "model name, and Groq account limits."
                    )


# ---------------- QUIZ GENERATOR ----------------

elif page == "📝 Quiz Generator":
    st.header("AI Quiz Generator")

    topic = st.text_input(
        "Enter Quiz Topic",
        value="Python Basics"
    )

    difficulty = st.selectbox(
        "Difficulty",
        ["Beginner", "Intermediate", "Advanced"]
    )

    count = st.slider(
        "Number of Questions",
        min_value=3,
        max_value=10,
        value=5
    )

    if st.button("Generate Quiz", type="primary"):
        quiz_prompt = f"""
Generate exactly {count} multiple-choice questions about {topic}
at {difficulty} level.

Return only valid JSON in this format:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": 0,
      "explanation": "Explanation"
    }}
  ]
}}

The answer must be an integer from 0 to 3 representing the correct
option's index. Each question must contain exactly four options.
"""

        try:
            with st.spinner("Generating quiz..."):
                raw = ask_ai([
                    {
                        "role": "system",
                        "content": "Create educational quizzes. Return valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": quiz_prompt
                    }
                ], temperature=0.2)

            raw = raw.strip()

            if raw.startswith("```"):
                lines = raw.splitlines()
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines).strip()
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()

            start = raw.find("{")
            end = raw.rfind("}")

            if start == -1 or end == -1:
                raise ValueError("The AI did not return valid JSON.")

            data = json.loads(raw[start:end + 1])
            questions = data["questions"]

            if not isinstance(questions, list) or not questions:
                raise ValueError("No quiz questions were generated.")

            for q in questions:
                if (
                    len(q["options"]) != 4
                    or q["answer"] not in [0, 1, 2, 3]
                ):
                    raise ValueError("The quiz response format is invalid.")

            st.session_state.current_quiz = {
                "topic": topic,
                "questions": questions[:count],
                "student_id": student_id
            }
            st.session_state.pop("quiz_submitted", None)

        except Exception as e:
            st.error(f"Quiz generation failed: {e}")

    quiz = st.session_state.get("current_quiz")

    if quiz and quiz["student_id"] == student_id:
        st.subheader(f"Quiz: {quiz['topic']}")

        if not st.session_state.get("quiz_submitted", False):
            with st.form("quiz_form"):
                answers = []

                for i, question in enumerate(quiz["questions"]):
                    st.markdown(
                        f"**Q{i + 1}. {question['question']}**"
                    )

                    selected = st.radio(
                        f"Select answer for question {i + 1}",
                        question["options"],
                        index=None,
                        key=f"answer_{student_id}_{i}"
                    )

                    answers.append(selected)

                submitted = st.form_submit_button("Submit Answers")

            if submitted:
                score = 0
                feedback = []

                for question, selected in zip(
                    quiz["questions"], answers
                ):
                    correct = question["options"][question["answer"]]
                    is_correct = selected == correct
                    score += int(is_correct)

                    feedback.append({
                        "question": question["question"],
                        "selected": selected or "Not answered",
                        "correct": correct,
                        "is_correct": is_correct,
                        "explanation": question.get(
                            "explanation", "Review this topic."
                        )
                    })

                st.session_state.quiz_score = score
                st.session_state.quiz_feedback = feedback
                st.session_state.quiz_submitted = True

                save_quiz_result(
                    student_id,
                    quiz["topic"],
                    score,
                    len(quiz["questions"])
                )

                st.rerun()

        else:
            score = st.session_state.quiz_score
            feedback = st.session_state.quiz_feedback
            total = len(feedback)

            st.metric("Your Score", f"{score}/{total}")
            st.progress(score / total if total else 0)

            for i, item in enumerate(feedback, 1):
                status = "Correct" if item["is_correct"] else "Incorrect"

                with st.expander(f"Question {i}: {status}"):
                    st.write(item["question"])
                    st.write("Your answer:", item["selected"])
                    st.write("Correct answer:", item["correct"])
                    st.write("Explanation:", item["explanation"])

            if st.button("Create Another Quiz"):
                for key in [
                    "current_quiz",
                    "quiz_submitted",
                    "quiz_score",
                    "quiz_feedback"
                ]:
                    st.session_state.pop(key, None)
                st.rerun()


# ---------------- STUDY PLANNER ----------------

elif page == "📅 Study Planner":
    st.header("Personalized Study Planner")

    goal = st.text_input(
        "What do you want to learn?",
        value=goals or "Learn Python programming"
    )

    days = st.slider("Plan Duration (days)", 3, 30, 7)
    hours = st.slider("Study Hours Per Day", 1, 8, 2)

    if st.button("Create Study Plan", type="primary"):
        prompt = f"""
Create a realistic {days}-day study plan.

Learning goal: {goal}
Student level: {level}
Explanation preference: {style}
Available study time: {hours} hours per day.

Give a day-by-day schedule with topics, practice, revision,
and checkpoints. Keep each day within the time limit.
Use clear Markdown.
"""

        try:
            with st.spinner("Creating your study plan..."):
                plan = ask_ai([
                    {
                        "role": "system",
                        "content": "You are a practical study planning assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ])

            st.markdown(plan)

            st.download_button(
                "Download Study Plan",
                data=plan,
                file_name="study_plan.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"Study plan generation failed: {e}")


# ---------------- LEARNING PROGRESS ----------------

elif page == "📊 Learning Progress":
    st.header("Learning Progress")

    with get_db() as conn:
        rows = conn.execute("""
            SELECT topic, score, total, created_at
            FROM quiz_results
            WHERE student_id=?
            ORDER BY id DESC
        """, (student_id,)).fetchall()

    if not rows:
        st.info("Complete a quiz to see your learning progress.")
    else:
        df = pd.DataFrame(
            rows,
            columns=["Topic", "Score", "Total Questions", "Completed At"]
        )

        df["Accuracy (%)"] = (
            df["Score"] / df["Total Questions"] * 100
        ).round(1)

        total_correct = int(df["Score"].sum())
        total_questions = int(df["Total Questions"].sum())
        accuracy = (
            total_correct / total_questions * 100
            if total_questions else 0
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Quizzes Completed", len(df))
        c2.metric("Questions Answered", total_questions)
        c3.metric("Overall Accuracy", f"{accuracy:.1f}%")

        st.dataframe(df, use_container_width=True)

        st.subheader("Quiz Accuracy")
        chart_df = df.iloc[::-1].reset_index(drop=True)
        chart_df["Quiz Number"] = range(1, len(chart_df) + 1)

        st.line_chart(
            chart_df.set_index("Quiz Number")["Accuracy (%)"]
        )

        st.subheader("Topics to Revise")
        topic_summary = df.groupby("Topic", as_index=False).agg(
            Correct=("Score", "sum"),
            Questions=("Total Questions", "sum")
        )
        topic_summary["Accuracy (%)"] = (
            topic_summary["Correct"] / topic_summary["Questions"] * 100
        ).round(1)

        st.dataframe(
            topic_summary.sort_values("Accuracy (%)"),
            use_container_width=True
        )


# ---------------- MEMORY SETTINGS ----------------

elif page == "🧠 Memory Settings":
    st.header("Conversational Memory")

    st.write(
        "The app stores recent tutor conversations, profile preferences, "
        "and quiz results in a local SQLite database."
    )

    history = load_messages(student_id, limit=50)
    st.metric("Messages in Recent History", len(history))

    if history:
        with st.expander("View Saved Conversation"):
            for message in history:
                st.markdown(f"**{message['role'].title()}**")
                st.write(message["content"])
    else:
        st.info("No saved conversation for this student.")

    st.warning(
        "Clearing conversation memory does not delete the student profile "
        "or quiz results."
    )

    confirm = st.checkbox(
        "I understand and want to clear my saved conversation."
    )

    if st.button("Clear Conversation Memory", disabled=not confirm):
        clear_messages(student_id)

        if st.session_state.get("active_student") == student_id:
            st.session_state.chat_history = []

        st.success("Conversation memory cleared.")
        st.rerun()


st.divider()
st.caption(
    "AI Personalized Learning Tutor | Streamlit + Groq + SQLite. "
    "AI-generated content can contain errors; verify important information."
)
