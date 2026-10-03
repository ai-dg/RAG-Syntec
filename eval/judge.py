from langchain_ollama import ChatOllama
from app.config import get_settings

JUDGE_PROMPT = """You are a strict fact-checker.
Decide whether EVERY claim in the ANSWER is supported by the CONTEXT.
Use only the CONTEXT, never outside knowledge. A claim that is plausible
but absent from the CONTEXT counts as NOT supported.

The first line of your reply must be exactly `VERDICT: YES` or `VERDICT: NO`.
Then, on the following lines, explain briefly why.

QUESTION: {question}

CONTEXT:
{context}

ANSWER:
{answer}
"""

def judge_faithfulness(
    question: str, answer: str, context: str, model: str = "llama3.2:3b"
) -> dict:

    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)

    llm = ChatOllama(
        model=model, base_url=get_settings().ollama_base_url, temperature=0.0
    )

    raw_output = llm.invoke(prompt).content

    lines = raw_output.strip().splitlines()

    first_line = lines[0].strip().upper() if lines else ""

    reasoning = "\n".join(lines[1:]).strip()

    if first_line == "VERDICT: YES":
        faithful = True
    elif first_line == "VERDICT: NO":
        faithful = False
    else:
        faithful = None
        reasoning = raw_output

    return {"faithful": faithful, "reasoning": reasoning}


def judge_agreement_rate(judge_verdicts: list[bool], human_labels: list[bool]) -> float:
    agreements = 0

    for judge_verdict, human_label in zip(judge_verdicts, human_labels):
        if judge_verdict == human_label:
            agreements += 1

    try:
        return agreements / len(human_labels)
    except ZeroDivisionError:
        return 0
