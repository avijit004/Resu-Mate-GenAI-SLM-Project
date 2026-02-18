from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from config import DEFAULT_EMBEDDING_MODEL, SIMILARITY_THRESHOLD, TOP_K_SECTIONS, USE_SEMANTIC_MATCHING
from semantic_matching import format_semantic_matches, get_top_matching_sections


def get_llm(model_name: str, temperature: float) -> OllamaLLM:
    return OllamaLLM(model=model_name, num_ctx=4096, temperature=temperature)


def extract_alignment_points(
    resume_text: str,
    job_desc: str,
    model_name: str,
    temperature: float,
    use_semantic_prefilter: bool = USE_SEMANTIC_MATCHING,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Tuple[str, str]:
    """
    Extract alignment points using hybrid approach: semantic matching + LLM refinement.
    
    Returns:
        Tuple of (formatted_alignment_points, semantic_matches_info)
    """
    semantic_matches_info = ""
    
    # Step 1: Semantic pre-filtering (if enabled)
    if use_semantic_prefilter:
        try:
            top_sections = get_top_matching_sections(
                resume_text,
                job_desc,
                top_k=TOP_K_SECTIONS,
                use_embeddings=True,
                embedding_model=embedding_model,
            )
            
            # Filter by threshold
            filtered_sections = [
                (sec, score) for sec, score in top_sections 
                if score >= SIMILARITY_THRESHOLD
            ]
            
            if filtered_sections:
                # Create a focused resume subset from top matches
                focused_resume = "\n\n".join([sec for sec, _ in filtered_sections])
                semantic_matches_info = format_semantic_matches(filtered_sections)
            else:
                focused_resume = resume_text
        except Exception:
            # Fallback: use full resume if semantic matching fails
            focused_resume = resume_text
    else:
        focused_resume = resume_text
    
    # Step 2: LLM-based extraction (refines semantic matches)
    template = """
    You are an expert career coach and professional writer.

    Your task is to extract the 5–10 MOST relevant, high‑impact points that show
    strong alignment between the candidate's RESUME and the JOB DESCRIPTION.

    Focus on:
    - Specific skills, tools, and technologies that match the role
    - Domain knowledge and problem spaces that overlap
    - Measurable achievements (with numbers) that are relevant
    - Leadership, ownership, and collaboration themes that matter for the role

    Return ONLY a concise markdown bullet list, one bullet per line,
    starting each line with "- " and nothing else before or after.

    RESUME (most relevant sections):
    {resume}

    JOB DESCRIPTION:
    {jd}
    """
    prompt = PromptTemplate.from_template(template)
    llm = get_llm(model_name=model_name, temperature=min(temperature, 0.7))
    chain = prompt | llm
    alignment_points = chain.invoke(
        {
            "resume": focused_resume,
            "jd": job_desc,
        }
    )
    
    return alignment_points, semantic_matches_info


def generate_stream(
    resume_text: str,
    job_desc: str,
    doc_type: str,
    tone: str,
    company: str,
    role: str,
    model_name: str,
    temperature: float,
    alignment_points: str,
):
    template = """
    You are an expert career coach and professional writer.

    Write a {tone} {doc_type} tailored for the following role and company:
    - ROLE: {role}
    - COMPANY: {company}

    Use the RESUME and JOB DESCRIPTION below to align skills, impact, and language.

    First, here are key matching skills and achievements between the resume and
    the job description. These should guide what you emphasize:

    MATCHING POINTS:
    {matches}

    When writing the {doc_type}:
    - Explicitly incorporate the strengths and matches listed above.
    - Include an early, clearly visible section titled "Key Highlights" that
      presents 3–7 concise bullet points recruiters can quickly skim, based on
      those matching points.
    - After the "Key Highlights" section, continue with the rest of the
      {doc_type} in natural, well-structured prose.

    RESUME:
    {resume}

    JOB DESCRIPTION:
    {jd}

    Guidelines:
    - Focus on concrete achievements and quantified impact where possible.
    - Mirror the key requirements and language from the job description.
    - Keep the length appropriate for a {doc_type}.
    - Make it easy to skim with clear structure and short paragraphs.

    Return ONLY the final {doc_type} text. Do not add explanations or notes.
    """
    prompt = PromptTemplate.from_template(template)
    llm = get_llm(model_name=model_name, temperature=temperature)
    chain = prompt | llm
    return chain.stream(
        {
            "resume": resume_text,
            "jd": job_desc,
            "doc_type": doc_type,
            "tone": tone,
            "company": company or "the company",
            "role": role or "the role",
            "matches": alignment_points
            or "- Strong alignment between experience and the role; focus on the most relevant skills and achievements.",
        }
    )

