import os
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY was not found. Add it to the .env file."
    )

client = Groq(api_key=api_key)


def summarize_meeting(
    transcript: str,
    language: str = "Russian"
):
    prompt = f"""Create a concise professional executive summary of this meeting.

TRANSCRIPT:
{transcript}

Rules:
1. Use ONLY information explicitly contained in the transcript.
2. Do not invent or correct names, places, numbers, dates, or facts.
3. OUTPUT LANGUAGE: {language}
   You MUST write the entire summary in {language}.
   Do not translate the summary into another language.
4. Summarize:
   - main topics discussed
   - important problems identified
   - important decisions and conclusions
   - important risks or concerns
5. Do NOT produce an action-item list.
6. Do NOT enumerate assignments such as:
   "1) person X must do Y by date Z".
7. Do NOT repeat detailed owner/deadline information.
   A separate validated action-item system handles this.
8. You may briefly mention that a decision or follow-up was agreed,
   but focus on the meeting outcome rather than task tracking.
9. Keep the summary concise and readable.
10. Preserve technical terminology from the transcript where possible.
11. Return valid JSON only.

Return:
{{
  "summary": "professional executive summary"
}}
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result["summary"]
