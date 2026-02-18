import streamlit as st
from ollama import ResponseError

from config import DEFAULT_MODEL, MODEL_OPTIONS, USE_SEMANTIC_MATCHING
from doc_processing import extract_text, is_ollama_online
from llm_service import extract_alignment_points, generate_stream
from pdf_export import create_pdf_from_text


st.set_page_config(page_title="Resu-Mate: Career Doc Generator", page_icon="📝", layout="wide")


def run_app() -> None:
    st.markdown(
        """
        <style>
          /* Tighter top padding + cleaner section headers */
          .block-container { padding-top: 1.4rem; }
          h1 { margin-bottom: 0.35rem; }
          h2, h3 { margin-top: 0.75rem; }
          /* Make sidebar feel less cramped */
          section[data-testid="stSidebar"] .block-container { padding-top: 1.1rem; }
          /* Nicer expander header weight */
          div[data-testid="stExpander"] details summary p { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Resu-Mate: Career Doc Generator")
    st.caption(
        "Generate a tailored **Cold Email** and **Cover Letter** from your resume + job description using a local Ollama model."
    )
    st.divider()

    if not is_ollama_online():
        st.error("⚠️ Ollama is offline. Please start the Ollama app on your computer.")
        st.stop()

    # --- Sidebar: Model & Style Controls ---
    with st.sidebar:
        st.header("⚙️ Settings")
        st.caption("Model, tone, and generation controls.")
        st.divider()

        st.success("Ollama: Online")

        labels = list(MODEL_OPTIONS.keys())
        default_label = DEFAULT_MODEL if DEFAULT_MODEL in labels else labels[0]
        selected_label = st.selectbox(
            "🤖 Ollama Model",
            labels,
            index=labels.index(st.session_state.get("model_label", default_label)),
            help=(
                "Choose between llama3.2:3b or Phi mini (phi3:mini). "
                "Make sure you've pulled it with `ollama pull`."
            ),
        )
        st.session_state["model_label"] = selected_label
        model_name = MODEL_OPTIONS[selected_label]

        st.caption(f"Using model tag: `{model_name}`")
        
        # Semantic matching toggle
        use_semantic = st.checkbox(
            "🔬 Enable semantic matching",
            value=USE_SEMANTIC_MATCHING,
            help="Use TF-IDF/embeddings + cosine similarity to pre-filter resume sections before SLM processing.",
        )
        st.session_state["use_semantic_matching"] = use_semantic
        
        st.divider()

        temperature = st.slider(
            "🎨 Creativity (Temperature)",
            min_value=0.0,
            max_value=1.5,
            value=0.7,
            step=0.05,
            help="Lower = safer & more conservative, higher = more creative.",
        )

        tone = st.selectbox(
            "✍️ Writing Tone",
            [
                "Professional & Formal",
                "Warm & Friendly",
                "Concise & Direct",
            ],
            help="Overall voice for the email and cover letter.",
        )

        st.divider()
        st.caption("Tip: Fill in company and role for better customization.")

    col1, col2 = st.columns([1.1, 1.9], gap="large")

    with col1:
        st.subheader("Input")
        st.caption("Step 1: Upload your resume and paste the job description.")
        st.divider()

        with st.form("inputs", clear_on_submit=False, border=False):
            uploaded_file = st.file_uploader(
                "Resume (PDF or DOCX)",
                type=["pdf", "docx"],
                help="Upload your resume in PDF or DOCX format.",
            )

            target_job = st.text_area(
                "Job Description",
                height=200,
                placeholder="Paste the full job description here…",
                help="Copy and paste the complete job posting.",
            )

            st.markdown("**Optional details (improves personalization)**")
            col_company, col_role = st.columns(2)
            with col_company:
                company = st.text_input(
                    "Company",
                    placeholder="e.g., Google",
                    value=st.session_state.get("company", ""),
                )
            with col_role:
                role = st.text_input(
                    "Role/Title",
                    placeholder="e.g., Software Engineer",
                    value=st.session_state.get("role", ""),
                )

            col_name, col_position = st.columns(2)
            with col_name:
                name = st.text_input(
                    "Your name",
                    placeholder="e.g., Avijit",
                    value=st.session_state.get("name", ""),
                )
            with col_position:
                position = st.text_input(
                    "Recipient position",
                    placeholder="e.g., Recruiter / Hiring Manager",
                    value=st.session_state.get("position", ""),
                )

            submitted = st.form_submit_button(
                "Generate drafts",
                type="primary",
                use_container_width=True,
            )

        # Persist optional fields
        st.session_state["company"] = company
        st.session_state["role"] = role
        st.session_state["name"] = name
        st.session_state["position"] = position

        if submitted:
            if uploaded_file and target_job and target_job.strip():
                resume_content = extract_text(uploaded_file)
                try:
                    with st.spinner("🔍 Finding key matching skills & achievements (using semantic matching + SLM)…"):
                        try:
                            alignment_points, semantic_info = extract_alignment_points(
                                resume_content,
                                target_job,
                                model_name,
                                temperature,
                                use_semantic_prefilter=st.session_state.get("use_semantic_matching", USE_SEMANTIC_MATCHING),
                            )
                        except Exception as e:
                            alignment_points = ""
                            semantic_info = ""
                        st.session_state.alignment_points = alignment_points
                        st.session_state.semantic_matches = semantic_info

                    with st.spinner("Generating Cold Email…"):
                        email_placeholder = st.empty()
                        full_email = ""
                        for chunk in generate_stream(
                            resume_content,
                            target_job,
                            "Cold Email",
                            tone,
                            company,
                            role,
                            model_name,
                            temperature,
                            st.session_state.get("alignment_points", ""),
                        ):
                            full_email += chunk
                            email_placeholder.markdown(full_email)
                        st.session_state.email_draft = full_email
                        st.session_state.email_edited = full_email

                    with st.spinner("Generating Cover Letter…"):
                        letter_placeholder = st.empty()
                        full_letter = ""
                        for chunk in generate_stream(
                            resume_content,
                            target_job,
                            "Cover Letter",
                            tone,
                            company,
                            role,
                            model_name,
                            temperature,
                            st.session_state.get("alignment_points", ""),
                        ):
                            full_letter += chunk
                            letter_placeholder.markdown(full_letter)
                        st.session_state.letter_draft = full_letter
                        st.session_state.letter_edited = full_letter

                    st.success("Done. Review, edit, and download on the right.")
                except ResponseError as e:
                    if e.status_code == 404:
                        st.error(
                            f"**Model `{model_name}` not found.** Install it by running:\n\n"
                            f"```\nollama pull {model_name}\n```"
                        )
                    else:
                        raise
            else:
                st.warning("Please upload a resume and paste a job description first.")

    with col2:
        st.subheader("Review & Edit")
        st.caption("Step 2: Edit drafts. Step 3: Export as PDF (cover letter) or text.")
        st.divider()

        if "email_draft" in st.session_state:
            if st.session_state.get("alignment_points"):
                with st.expander("🎯 Key matching skills & achievements (SLM-extracted)", expanded=False):
                    st.markdown(st.session_state.alignment_points)
            
            if st.session_state.get("semantic_matches") and USE_SEMANTIC_MATCHING:
                with st.expander("📊 Semantic similarity scores (TF-IDF/Embeddings)", expanded=False):
                    st.markdown(st.session_state.semantic_matches)
                    st.caption("💡 Top matching resume sections ranked by cosine similarity with job description.")

            tab1, tab2 = st.tabs(["📧 Cold Email", "📄 Cover Letter"])

            with tab1:
                if "email_edited" not in st.session_state:
                    st.session_state.email_edited = st.session_state.email_draft

                updated_email = st.text_area(
                    "Edit your cold email:",
                    value=st.session_state.email_edited,
                    height=350,
                    help="Make any edits you want, then download when ready.",
                )
                st.session_state.email_edited = updated_email

                word_count = len(updated_email.split())
                st.caption(f"📊 Word count: {word_count}")

                with st.expander("👁️ Preview"):
                    st.markdown(updated_email)

                st.download_button(
                    "Download Cold Email (TXT)",
                    data=updated_email.encode("utf-8"),
                    file_name="cold_email.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with tab2:
                if "letter_edited" not in st.session_state:
                    st.session_state.letter_edited = st.session_state.letter_draft

                updated_letter = st.text_area(
                    "Edit your cover letter:",
                    value=st.session_state.letter_edited,
                    height=400,
                    help="Make any edits you want, then download when ready.",
                )
                st.session_state.letter_edited = updated_letter

                word_count = len(updated_letter.split())
                col_info, col_download = st.columns([2, 1])
                with col_info:
                    st.caption(f"📊 Word count: {word_count}")
                with col_download:
                    company_name = (
                        st.session_state.get("company", "").replace(" ", "_")
                        if st.session_state.get("company")
                        else "draft"
                    )
                    pdf_data = create_pdf_from_text(updated_letter)
                    st.download_button(
                        "💾 Download Cover Letter (PDF)",
                        data=pdf_data,
                        file_name=f"cover_letter_{company_name}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                    )

                with st.expander("👁️ Preview"):
                    st.markdown(updated_letter)

                st.download_button(
                    "Download Cover Letter (TXT)",
                    data=updated_letter.encode("utf-8"),
                    file_name=f"cover_letter_{company_name}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        else:
            st.info(
                "Upload your resume and job description, then click **Generate drafts** to get started."
            )
            st.markdown(
                """
                **How it works:**
                1. Upload your resume (PDF/DOCX)
                2. Paste the job description
                3. Pick a model + tone (sidebar)
                4. Generate drafts, then edit
                5. Download the cover letter as PDF
                """
            )

