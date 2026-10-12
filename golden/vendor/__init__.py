"""Code copied from Hyde-Inc/trace-mining (commit 4f4b8a9) so this repo runs on its own.

    llm.py           trace_mining/llm.py (unchanged): OpenAI-compatible chat client with retries
    harness.py       trace_mining/blocks/harness.py (unchanged): Turn, turn_line, the trace turn parser
    pseudo.py        trace_mining/blocks/pseudo.py (unchanged): pseudonymizer for ids, emails, phones
    minhash.py       MinHasher from trace_mining/common.py
    segmentation.py  DEFINITION, LABEL_SYSTEM and label_text from trace_mining/blocks/segment.py

Keep them in sync by hand if the originals change; do not edit them here.
"""
