from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

from collections import defaultdict
import pandas as pd
import os
import re
import numpy as np
import math

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm

from policy_framework_mapper import *

# Skywork Reward Model Evaluator
# ===============================
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

# --- Load the Skywork reward model ---
model_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="cpu",
    num_labels=1,  # reward models output a single scalar
    verbose=True,
)

# --- Token IDs for binary "Yes"/"No" ---
YES_TOKEN_ID = tokenizer.encode("Yes", add_special_tokens=False)[0]
NO_TOKEN_ID  = tokenizer.encode("No",  add_special_tokens=False)[0]

# Message construction aligned to Skywork reward model format
# ============================================================

POLICY_PRINCIPLE = """
You are evaluating a single policy clause. The clause may have multiple subsections.
You will be provided with the applicable framework, and the controls and subcontrols
that the clause is intended to address. You may also be supplied with the organization's
context (industry, size, risk tolerance, key assets, etc.) when available.
If this context is not provided, you should evaluate the clause based solely on the framework requirements.

Evaluate how well the clause satisfies the following criteria:
1. Be precise, enforceable, and free of ambiguity.
2. Correctly align with the stated framework requirements (e.g., NIST CSF, SOC 2, ISO 27001) when referenced.
3. Reflect the organization's ontology (industry, size, risk tolerance, key assets, etc.) when such context is provided.
4. Avoid hallucinated frameworks, controls, or obligations.
5. Maintain a professional, directive tone appropriate for a formal corporate policy.

Special evaluation rules:
- **If no specific controls or subcontrols are provided**:  
  Do *not* penalize the clause for lacking explicit control references.  
  Instead, focus on clarity, correctness, and appropriateness for its section type.

- **If the section is an introductory or structural section**
  (for example *Purpose*, *Scope*, *Definitions*, *Roles & Responsibilities*):  
  Evaluate it on clarity of intent, relevance to the policy and framework theme, and professional tone.  
  Such sections are not expected to contain control implementation details or enforceable requirements.

- **Only assign low scores (<50)** when the clause is clearly off-topic,
  factually incorrect, internally contradictory, or so vague it fails to communicate intent.

Assign a numeric score from 0 to 100, where:
- 0 = completely non-compliant with the principle,
- 50 = partially meets requirements but contains material gaps or ambiguities, and
- 100 = fully compliant and exemplary in clarity, alignment, and tone.

Your goal is to output a single numeric score reflecting how well the clause adheres to these principles.
Do not include any explanations or commentary — only return the numeric score.
""".strip()

def skywork_raw_reward_for_clause(clause_text: str,
                                  subcontrol_text: str) -> float:
    """
    Returns the raw Skywork reward (unbounded scalar) for how well
    `clause_text` performs as an 'answer' under the constraint of `subcontrol_text`.
    """
    messages = [
        {
            "role": "user",
            "content": (
                "You are evaluating the next assistant message, which is a proposed policy clause. "
                "It may include multiple subsections. Use the following principle and framework context to judge it.\n\n"
                f"{POLICY_PRINCIPLE}\n\n"
                "Framework sub-control requirement:\n"
                f"{subcontrol_text}\n\n"
                "Reward higher if it is:\n"
                "- Precise and enforceable (no vague language)\n"
                "- Directly aligned with the sub-control scope and obligations\n"
                "- Complete on key points (who, what, when, how)\n"
                "- Free of contradictions or hallucinated requirements\n"
                "- Professional and directive in tone\n"
                "Penalize if it is vague, off-scope, incorrect, or incomplete."
            ),
        },
        {
            # This is the thing being judged
            "role": "assistant",
            "content": clause_text.strip(),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Model card note: avoid duplicate BOS if present.
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        out = model(**inputs)
        raw = out.logits[0][0].item()

    return float(raw)

def skywork_clause_framework_score(clause_text: str,
                                   subcontrol_text: str) -> float:
    """
    Returns a 0-100 adherence score using Skywork RM as backbone.
    """
    raw = skywork_raw_reward_for_clause(clause_text, subcontrol_text)
    # Logistic squashing: maps (-inf, +inf) -> (0, 1)
    prob = 1.0 / (1.0 + math.exp(-raw))

    # Scale to 0-100
    return float(prob * 100.0)


def apply_judge(
    df,
    prompt_col: str='clause_full_text',
    framework_col: str='source_framework',
    policy_name_col: str="policy_title",
    section_num_col: str="clause_section_number",
    section_title_col: str="section_title",
    output_col: str="output"
    ):

    scores = []
    pbar = tqdm(
        total=len(df),
        desc="Evaluating policy clauses",
        unit="clause",
        leave=True
    )

    for index, row in df.iterrows():
        policy_name = row.get(policy_name_col)
        section_num = row.get(section_num_col, "")
        section_title = row.get(section_title_col, "")
        framework = row.get(framework_col)
        clause_text = row.get(prompt_col)

        # Update progress bar message dynamically
        pbar.set_postfix({
            "Policy": str(policy_name)[:25],
            "Section": str(section_num),
            "Framework": str(framework)
        })

        section_context = f"Policy: {policy_name}\nSection {section_num}: {section_title}".strip()
        clause_block = f"{section_context}\n\nClause to evaluate:\n{clause_text.strip()}"

        # # Build inputs and run scoring
        framework_data = get_framework_context_for_policy(policy_name, framework)
        score = skywork_clause_framework_score(clause_block, framework_data)
        scores.append(score)
        pbar.update(1)

    pbar.close()

    df[output_col] = scores
    return df

