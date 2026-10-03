"""
LangSmith Evaluator Agent — 7 Structured Test Cases with Exact Error Detection
=============================================================================
A clean, powerful Streamlit application that generates EXACTLY 7 diverse,
structured test cases with pinpoint error detection from a Question and
Expected Answer, and provides in-card evaluation with full LangSmith tracing.
"""

import json
import os
import time
from enum import Enum
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Load environment variables
# Load environment variables
load_dotenv(override=True)


# -----------------------------------------------------------------------------
# 1. Pydantic Schemas for 7 Structured Test Cases & Evaluation
# -----------------------------------------------------------------------------
class VerdictEnum(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


class ErrorStatusEnum(str, Enum):
    NO_ERROR = "NO ERROR"
    ERROR = "ERROR"


class GeneratedTestCase(BaseModel):
    test_case_id: str = Field(description="Unique ID, e.g., TC-001, TC-002, ..., TC-007")
    test_case_type: str = Field(
        description="Exactly one of: CORRECT, CORRECT PARAPHRASED, PARTIALLY CORRECT, INCOMPLETE, WRONG, IRRELEVANT, MISLEADING"
    )
    question: str = Field(description="The original question")
    expected_answer: str = Field(description="The reference expected answer")
    sample_answer: str = Field(description="The simulated AI-generated answer to evaluate")
    expected_verdict: VerdictEnum = Field(description="Expected verdict: PASS, PARTIAL, or FAIL")
    error_status: ErrorStatusEnum = Field(description="NO ERROR for correct answers, ERROR for flawed answers")
    error_location: str = Field(
        description="Exact incorrect phrase, missing concept, 'None' for correct answers, or 'Entire answer.'"
    )
    error_explanation: str = Field(
        description="Short explanation of the error or why the answer is correct"
    )
    corrected_answer: str = Field(
        description="The correct information, or 'Not required' if the answer is already correct"
    )


class GeneratedTestSuite(BaseModel):
    test_cases: List[GeneratedTestCase] = Field(
        description="A list containing EXACTLY 7 test cases representing the 7 categories in order."
    )


class EvaluationResult(BaseModel):
    verdict: VerdictEnum = Field(
        description="Overall evaluation verdict: PASS (meets criteria), PARTIAL (partially meets criteria), or FAIL (does not meet criteria)."
    )
    correctness_assessment: str = Field(
        description="Assessment of whether the generated answer is factually correct compared to the reference answer. Semantic equivalence is allowed."
    )
    relevance_assessment: str = Field(
        description="Assessment of whether the generated answer directly and fully addresses the user question."
    )
    faithfulness_assessment: str = Field(
        description="Assessment of whether the generated answer stays true to the reference answer without inventing unsupported claims or hallucinations."
    )
    detected_error_location: str = Field(
        description="Exact sentence, phrase, or missing concept where the flaw occurs, or 'None' if the answer is accurate."
    )
    detected_error_explanation: str = Field(
        description="Clear explanation of the detected flaw or confirmation of accuracy."
    )
    explanation: str = Field(
        description="A concise summary explanation justifying the overall verdict."
    )
    improvement_suggestions: str = Field(
        description="Actionable feedback on how the AI-generated answer can be improved."
    )


# -----------------------------------------------------------------------------
# 2. LLM Chains (7-Case Generator & Evaluator)
# -----------------------------------------------------------------------------
def get_generator_chain(provider: str, api_key: str, model_name: str):
    """
    Constructs the LangChain chain that generates EXACTLY 7 distinct structured test cases.
    """
    from langchain_core.prompts import ChatPromptTemplate

    system_prompt = """You are an expert AI QA and Evaluation Engineer. Given a Question and Expected Reference Answer, you MUST generate EXACTLY 7 distinct structured test cases.

You must create ONE test case for EACH of the following 7 categories in order:

1. **CORRECT**
   - Sample Answer: A fully correct, direct answer.
   - expected_verdict: PASS
   - error_status: NO ERROR
   - error_location: "None"
   - error_explanation: "The answer correctly addresses the question and agrees with the reference answer."
   - corrected_answer: "Not required"

2. **CORRECT PARAPHRASED**
   - Sample Answer: A fully correct answer using different phrasing/vocabulary.
   - expected_verdict: PASS
   - error_status: NO ERROR
   - error_location: "None"
   - error_explanation: "The answer correctly conveys the key information using alternative phrasing."
   - corrected_answer: "Not required"

3. **PARTIALLY CORRECT**
   - Sample Answer: Contains some correct information but includes a minor factual inaccuracy.
   - expected_verdict: PARTIAL
   - error_status: ERROR
   - error_location: Quote the specific flawed phrase or incomplete statement.
   - error_explanation: Explain what is incorrect or partially inaccurate.
   - corrected_answer: Provide the corrected statement.

4. **INCOMPLETE**
   - Sample Answer: Gives a high-level idea but omits crucial core concepts from the reference answer.
   - expected_verdict: PARTIAL
   - error_status: ERROR
   - error_location: "Missing details: [list specific missing key concepts]"
   - error_explanation: "Explain what information is missing and why it matters."
   - corrected_answer: Provide the complete answer.

5. **WRONG**
   - Sample Answer: Contains a clear factual or technical error/contradiction.
   - expected_verdict: FAIL
   - error_status: ERROR
   - error_location: Quote the exact incorrect sentence or phrase from the sample answer.
   - error_explanation: Explain why that specific statement is factually incorrect.
   - corrected_answer: Provide the correct factual information.

6. **IRRELEVANT**
   - Sample Answer: Answers a completely different question or discusses an unrelated topic.
   - expected_verdict: FAIL
   - error_status: ERROR
   - error_location: "Entire answer."
   - error_explanation: "Explain why the answer does not address the question."
   - corrected_answer: Provide the expected answer to the original question.

7. **MISLEADING**
   - Sample Answer: Blends truthful context with a significant false or deceptive claim.
   - expected_verdict: FAIL
   - error_status: ERROR
   - error_location: Quote the specific false or misleading phrase.
   - error_explanation: Separate the correct information from the incorrect claim.
   - corrected_answer: Provide the corrected answer without the misleading claim.

CRITICAL INSTRUCTIONS:
- You MUST return a list of EXACTLY 7 test cases (TC-001 through TC-007).
- Do NOT return fewer than 7 test cases.
- Never invent an error for CORRECT or CORRECT PARAPHRASED answers.
- Ensure every field is populated accurately."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        (
            "human",
            """Please generate the 7 structured test cases for:

[QUESTION]
{question}

[EXPECTED ANSWER]
{expected_answer}

Return all 7 test cases.""",
        ),
    ])

    if provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            google_api_key=api_key,
            max_retries=3,
        )
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            api_key=api_key,
            max_retries=3,
        )

    return prompt | llm.with_structured_output(GeneratedTestSuite)


def get_evaluator_chain(provider: str, api_key: str, model_name: str):
    """
    Constructs the LangChain evaluation chain with structured output and error detection.
    """
    from langchain_core.prompts import ChatPromptTemplate

    system_prompt = """You are an expert AI Evaluation Judge. Your job is to objectively evaluate an AI-generated answer against a trusted Reference Answer and a User Question, and identify the exact location of any error.

### Evaluation Criteria:
1. **Correctness**: Does the AI answer convey the correct facts as stated in the reference answer? Semantic equivalence should be treated as correct (exact wording is NOT required).
2. **Relevance**: Does the AI answer directly and helpfully address the user's specific question?
3. **Faithfulness**: Does the AI answer remain strictly grounded in the facts of the reference answer, without hallucinating, contradicting, or inventing false details?

### Error Location Rules:
- If the answer is correct: State detected_error_location as "None" and confirm accuracy in detected_error_explanation. Do NOT invent errors.
- If the answer is incomplete: State detected_error_location as "Missing details: [list missing items]" and explain what is omitted.
- If the answer contains a factual error: State detected_error_location as "The phrase '[exact text]'" or the specific sentence, and explain why it is wrong.
- If the answer is irrelevant: State detected_error_location as "Entire answer." and explain why.

### Verdict Rubric:
- **PASS**: The answer is factually accurate according to the reference, relevant to the question, and contains no false hallucinations.
- **PARTIAL**: The answer captures some correct key points but misses essential details, contains minor inaccuracies, or partially deviates from the prompt.
- **FAIL**: The answer is factually wrong, contradicts the reference answer, completely fails to answer the question, or is entirely irrelevant.

Be objective, constructive, and concise."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        (
            "human",
            """Please evaluate the following:

[USER QUESTION]
{question}

[REFERENCE ANSWER (Ground Truth)]
{reference_answer}

[AI-GENERATED ANSWER (To Evaluate)]
{generated_answer}

Provide your structured evaluation including detected error location and explanation.""",
        ),
    ])

    if provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            google_api_key=api_key,
            max_retries=3,
        )
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            api_key=api_key,
            max_retries=3,
        )

    return prompt | llm.with_structured_output(EvaluationResult)


def run_generation(
    question: str,
    expected_answer: str,
    provider: str,
    api_key: str,
    model_name: str,
) -> GeneratedTestSuite:
    """
    Generates exactly 7 test cases with retry validation and LangSmith tracing.
    """
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "true"
        if not os.getenv("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = "LangSmith-Evaluator-Agent"

    generator_chain = get_generator_chain(provider, api_key, model_name)

    config = {
        "run_name": "Generate_7_Test_Cases",
        "tags": ["test-case-generator", "streamlit-ui", provider.lower().replace(" ", "-")],
        "metadata": {
            "provider": provider,
            "generator_model": model_name,
            "task": "generate_7_structured_test_cases",
        },
    }

    result: GeneratedTestSuite = generator_chain.invoke(
        {
            "question": question,
            "expected_answer": expected_answer,
        },
        config=config,
    )

    # Ensure IDs are standardized TC-001 to TC-007
    for idx, tc in enumerate(result.test_cases):
        tc.test_case_id = f"TC-{idx+1:03d}"

    return result


def run_evaluation(
    question: str,
    reference_answer: str,
    generated_answer: str,
    provider: str,
    api_key: str,
    model_name: str,
    test_case_id: Optional[str] = None,
) -> EvaluationResult:
    """
    Executes the evaluation chain with LangSmith tracing enabled.
    """
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "true"
        if not os.getenv("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = "LangSmith-Evaluator-Agent"

    evaluator_chain = get_evaluator_chain(provider, api_key, model_name)

    tags = ["evaluator-agent", "streamlit-ui", provider.lower().replace(" ", "-")]
    if test_case_id:
        tags.append(test_case_id)

    config = {
        "run_name": "Evaluate_Test_Case",
        "tags": tags,
        "metadata": {
            "provider": provider,
            "evaluator_model": model_name,
            "evaluator_type": "LLM_as_a_judge",
            "test_case_id": test_case_id or "adhoc",
        },
    }

    result = evaluator_chain.invoke(
        {
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
        },
        config=config,
    )
    return result


# -----------------------------------------------------------------------------
# 3. Streamlit User Interface
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="LangSmith Evaluator Agent — 7 Test Cases Generator",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧪 LangSmith Evaluator Agent")
    st.caption(
        "Generate **exactly 7 structured test cases** with **exact error detection** from a Question and Expected Answer."
    )

    # ---------------- Sidebar: Settings & Configuration ----------------
    with st.sidebar:
        st.header("⚙️ Evaluation Engine")

        env_gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        env_openai_key = os.getenv("OPENAI_API_KEY", "")
        env_langsmith_key = os.getenv("LANGSMITH_API_KEY", "")
        env_project = os.getenv("LANGSMITH_PROJECT", "LangSmith-Evaluator-Agent")

        provider = st.selectbox(
            "LLM Provider",
            options=["Google Gemini", "OpenAI"],
            index=0 if env_gemini_key else 1,
            help="Provider used for generation and evaluation.",
        )

        st.subheader("1. API Key")
        if provider == "Google Gemini":
            api_key_input = st.text_input(
                "Google Gemini API Key",
                value=env_gemini_key,
                type="password",
                placeholder="AIzaSy... or AQ...",
                help="Set in .env (GOOGLE_API_KEY) or enter here.",
            )
            model_options = [
                "gemini-3.1-flash-lite",
                "gemini-3.7-flash",
                "gemini-3.8-flash",
                "gemini-3.5-flash",
                "gemini-flash-latest",
                "gemini-pro-latest",
            ]
        else:
            api_key_input = st.text_input(
                "OpenAI API Key",
                value=env_openai_key,
                type="password",
                placeholder="sk-...",
                help="Set in .env (OPENAI_API_KEY) or enter here.",
            )
            model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

        st.subheader("2. Model")
        model_name = st.selectbox("Select Model", options=model_options, index=0)

        st.subheader("3. LangSmith Tracing")
        langsmith_key_input = st.text_input(
            "LangSmith API Key",
            value=env_langsmith_key,
            type="password",
            placeholder="lsv2_pt_...",
            help="Set in .env (LANGSMITH_API_KEY) or enter here.",
        )
        project_name = st.text_input("Project Name", value=env_project)

        if langsmith_key_input:
            os.environ["LANGSMITH_API_KEY"] = langsmith_key_input
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_PROJECT"] = project_name
            st.success("🟢 LangSmith Tracing Active", icon="📡")
        else:
            st.warning("⚪ LangSmith Key Not Set", icon="⚠️")

        st.markdown("---")
        st.markdown("### 💡 Example Prompt Presets")
        example_presets = {
            "Select an example...": {"q": "", "a": ""},
            "Spring Boot Auto-Configuration": {
                "q": "What is Spring Boot auto-configuration?",
                "a": "Spring Boot auto-configuration configures application components automatically based on the classpath, existing beans, and configuration properties.",
            },
            "Spring Boot @SpringBootApplication": {
                "q": "What is the purpose of the @SpringBootApplication annotation?",
                "a": "@SpringBootApplication is a convenience annotation that combines @SpringBootConfiguration, @EnableAutoConfiguration, and @ComponentScan. It marks the main configuration class, enables auto-configuration, and allows component scanning from its package and subpackages.",
            },
            "Spring Boot Starters": {
                "q": "What are Spring Boot Starter dependencies?",
                "a": "Spring Boot Starters are convenient dependency descriptors that group commonly used libraries for a particular purpose, such as spring-boot-starter-web. They simplify dependency management by avoiding the need to add each library individually.",
            },
        }

        selected_example = st.selectbox("Load Preset", list(example_presets.keys()))
        if selected_example != "Select an example...":
            st.session_state["input_q"] = example_presets[selected_example]["q"]
            st.session_state["input_ans"] = example_presets[selected_example]["a"]

    # ---------------- Main Form: ONLY 2 Inputs ----------------
    st.subheader("📝 Input Question & Expected Answer")

    question_input = st.text_area(
        "1. Question",
        value=st.session_state.get("input_q", ""),
        placeholder="e.g., What is Spring Boot auto-configuration?",
        height=100,
        key="main_question_input",
    )

    expected_answer_input = st.text_area(
        "2. Expected Answer",
        value=st.session_state.get("input_ans", ""),
        placeholder="e.g., Spring Boot auto-configuration configures application components automatically based on the classpath, existing beans, and configuration properties.",
        height=120,
        key="main_expected_answer_input",
    )

    generate_btn = st.button("🚀 Generate Test Cases", type="primary", use_container_width=True)

    # ---------------- Test Case Generation Logic ----------------
    if generate_btn:
        if not api_key_input.strip():
            st.error(f"❌ Please enter a **{provider} API Key** in the sidebar.")
            return

        if not question_input.strip():
            st.warning("⚠️ Please provide a **Question**.")
            return

        if not expected_answer_input.strip():
            st.warning("⚠️ Please provide an **Expected Answer**.")
            return

        with st.spinner(f"✨ Generating exactly 7 structured test cases with error detection using {model_name}..."):
            start_gen = time.time()
            try:
                test_suite: GeneratedTestSuite = run_generation(
                    question=question_input.strip(),
                    expected_answer=expected_answer_input.strip(),
                    provider=provider,
                    api_key=api_key_input.strip(),
                    model_name=model_name,
                )
                gen_duration = round(time.time() - start_gen, 2)
                st.session_state["current_question"] = question_input.strip()
                st.session_state["current_expected_answer"] = expected_answer_input.strip()
                st.session_state["generated_test_cases"] = [tc.model_dump() for tc in test_suite.test_cases]
                st.session_state["gen_duration"] = gen_duration
                st.session_state["eval_results"] = {}  # Reset evaluation cache
                st.success(f"✅ Generated **{len(test_suite.test_cases)} structured test cases** in {gen_duration}s!")
            except Exception as e:
                err_text = str(e)
                if api_key_input and api_key_input in err_text:
                    err_text = err_text.replace(api_key_input, "[REDACTED_API_KEY]")
                st.error(f"❌ Generation failed: {err_text}")
                return

    # ---------------- Display Generated Test Cases in Structured Cards ----------------
    if "generated_test_cases" in st.session_state and st.session_state["generated_test_cases"]:
        st.markdown("---")
        st.subheader("📋 Generated Test Cases (7 Structured Cases)")

        # Summary Header
        st.markdown(f"**Question:** {st.session_state.get('current_question', '')}")
        with st.expander("📖 View Expected Ground Truth Answer", expanded=False):
            st.info(st.session_state.get("current_expected_answer", ""))

        test_cases = st.session_state["generated_test_cases"]
        st.caption(f"Displaying **{len(test_cases)} test cases**:")

        for idx, tc in enumerate(test_cases):
            tc_id = tc.get("test_case_id", f"TC-{idx+1:03d}")
            tc_type = tc.get("test_case_type", "TEST CASE")
            exp_verdict = tc.get("expected_verdict", "PASS")
            err_status = tc.get("error_status", "NO ERROR")

            # Badges
            v_badge = "🟢 PASS" if exp_verdict == "PASS" else ("🟡 PARTIAL" if exp_verdict == "PARTIAL" else "🔴 FAIL")
            s_badge = "🟢 NO ERROR" if err_status == "NO ERROR" else "🔴 ERROR"

            with st.expander(f"**TEST CASE {tc_id}** &nbsp;|&nbsp; Type: **{tc_type}** &nbsp;|&nbsp; Expected: **{v_badge}**", expanded=True):
                # Structured Layout - No Paragraph Dumps
                st.markdown(f"### TEST CASE {tc_id}")
                st.markdown(f"**Type:** `{tc_type}`")
                st.markdown(f"**Question:**\n{tc.get('question', st.session_state.get('current_question', ''))}")
                st.markdown(f"**Expected Answer:**\n{tc.get('expected_answer', st.session_state.get('current_expected_answer', ''))}")

                st.markdown("---")
                st.markdown("#### **Sample AI Answer:**")
                st.info(tc.get("sample_answer", ""))

                # Metrics / Status Table
                col_v, col_s = st.columns(2)
                with col_v:
                    st.markdown(f"**Expected Verdict:** {v_badge}")
                with col_s:
                    st.markdown(f"**Error Status:** {s_badge}")

                st.markdown(f"**Error Location:**")
                err_loc = tc.get("error_location", "None")
                if err_loc == "None" or "no error" in err_loc.lower():
                    st.success(err_loc)
                else:
                    st.warning(err_loc)

                st.markdown(f"**Why:**\n{tc.get('error_explanation', '')}")
                st.markdown(f"**Corrected Answer:**\n{tc.get('corrected_answer', 'Not required')}")

                # ---------------- In-Card Evaluation Button ----------------
                st.markdown("---")
                eval_btn_key = f"btn_eval_card_{tc_id}_{idx}"
                if st.button(f"🚀 Evaluate {tc_id}", key=eval_btn_key, type="secondary"):
                    if not api_key_input.strip():
                        st.error(f"❌ Please enter a **{provider} API Key** in the sidebar before evaluating.")
                    else:
                        with st.spinner(f"🔍 Evaluator evaluating {tc_id} against rubric..."):
                            start_eval = time.time()
                            try:
                                eval_res: EvaluationResult = run_evaluation(
                                    question=tc.get("question", st.session_state.get("current_question", "")),
                                    reference_answer=tc.get("expected_answer", st.session_state.get("current_expected_answer", "")),
                                    generated_answer=tc.get("sample_answer", ""),
                                    provider=provider,
                                    api_key=api_key_input.strip(),
                                    model_name=model_name,
                                    test_case_id=tc_id,
                                )
                                eval_dur = round(time.time() - start_eval, 2)
                                if "eval_results" not in st.session_state:
                                    st.session_state["eval_results"] = {}
                                st.session_state["eval_results"][tc_id] = {
                                    "result": eval_res.model_dump(),
                                    "duration": eval_dur,
                                }
                            except Exception as e:
                                err_eval_text = str(e)
                                if api_key_input and api_key_input in err_eval_text:
                                    err_eval_text = err_eval_text.replace(api_key_input, "[REDACTED_API_KEY]")
                                st.error(f"❌ Evaluation failed: {err_eval_text}")

                # ---------------- Display Actual Evaluation Results ----------------
                if "eval_results" in st.session_state and tc_id in st.session_state["eval_results"]:
                    cached_eval = st.session_state["eval_results"][tc_id]
                    res_data = cached_eval["result"]
                    actual_v = res_data["verdict"]
                    eval_dur = cached_eval["duration"]

                    st.markdown("#### 📊 Evaluator Verdict & Inspection:")

                    # Compare Actual vs Expected Verdict
                    matches = (actual_v == exp_verdict)
                    alignment_badge = "✅ Matches Expected Verdict" if matches else "⚠️ Verdict Diverged from Expected"

                    if actual_v == "PASS":
                        st.success(f"### Actual Verdict: **PASS** ✅ &nbsp;&nbsp; `({alignment_badge})`", icon="✅")
                    elif actual_v == "PARTIAL":
                        st.warning(f"### Actual Verdict: **PARTIAL** ⚠️ &nbsp;&nbsp; `({alignment_badge})`", icon="⚠️")
                    else:
                        st.error(f"### Actual Verdict: **FAIL** ❌ &nbsp;&nbsp; `({alignment_badge})`", icon="❌")

                    st.caption(f"⏱️ Evaluated in **{eval_dur}s** using **{provider} ({model_name})**")

                    # 3-Dimension Breakdown
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("🎯 **Correctness:**")
                        st.write(res_data["correctness_assessment"])
                    with c2:
                        st.markdown("🔍 **Relevance:**")
                        st.write(res_data["relevance_assessment"])
                    with c3:
                        st.markdown("🛡️ **Faithfulness:**")
                        st.write(res_data["faithfulness_assessment"])

                    # Detected Error Location & Explanation
                    det_loc_col, det_exp_col = st.columns(2)
                    with det_loc_col:
                        st.markdown("🎯 **Actual Evaluator Detected Error Location:**")
                        det_loc = res_data.get("detected_error_location", "None")
                        if det_loc == "None" or "no error" in det_loc.lower():
                            st.success(det_loc)
                        else:
                            st.warning(det_loc)

                    with det_exp_col:
                        st.markdown("🔍 **Actual Evaluator Detected Error Explanation:**")
                        st.write(res_data.get("detected_error_explanation", ""))

                    st.markdown("📋 **Summary Explanation:**")
                    st.info(res_data["explanation"])

                    st.markdown("💡 **Improvement Suggestions:**")
                    st.write(res_data["improvement_suggestions"])


if __name__ == "__main__":
    main()
