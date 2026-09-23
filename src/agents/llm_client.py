# ============================================================
# AutoDS - Gemini LLM Client
# Free-Tier Compatible Version
# ============================================================

import json
import os

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite"
)


GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/openai/"
)


# ============================================================
# AUTODS SYSTEM RULES
# ============================================================

AUTODS_RULES = """
You are one specialist agent inside an Autonomous Data
Scientist system.

You must follow these rules:

1. Use ONLY facts supplied in the payload.

2. Never invent dataset statistics.

3. Never invent model accuracy, F1, ROC-AUC,
   RMSE, R2, precision, recall, or any other metric.

4. Never invent feature importance.

5. Never claim a pattern exists unless the supplied
   deterministic data supports it.

6. Clearly distinguish:
   - verified facts
   - observations
   - recommendations

7. If required information is unavailable,
   explicitly state that it is unavailable.

8. Never change numerical values supplied by Python.

9. Recommendations are future actions and must not
   be presented as completed experiments.

10. Be concise, technical, and data-science focused.
"""


# ============================================================
# CREATE GEMINI CLIENT
# ============================================================

def get_client():

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY was not found.\n"
            "Please check your .env file."
        )

    client = OpenAI(

        api_key=api_key,

        base_url=GEMINI_BASE_URL
    )

    return client


# ============================================================
# STRUCTURED GEMINI REQUEST
# ============================================================

def ask_structured(
    output_schema,
    system_prompt: str,
    payload: dict
):
    """
    Send verified AutoDS information to Gemini and force
    the response into the supplied Pydantic schema.
    """

    client = get_client()


    # --------------------------------------------------------
    # Combine AutoDS safety instructions with
    # the individual agent prompt
    # --------------------------------------------------------

    full_system_prompt = (

        AUTODS_RULES

        +

        "\n\n"

        +

        system_prompt
    )


    # --------------------------------------------------------
    # Convert Python/Pandas information safely into JSON
    # --------------------------------------------------------

    user_content = json.dumps(

        payload,

        indent=2,

        default=str
    )


    # --------------------------------------------------------
    # Gemini structured output
    # through OpenAI-compatible API
    # --------------------------------------------------------

    completion = (
        client.beta.chat.completions.parse(

            model=MODEL_NAME,

            messages=[

                {
                    "role": "system",
                    "content": full_system_prompt
                },

                {
                    "role": "user",
                    "content": user_content
                }

            ],

            response_format=output_schema
        )
    )


    # --------------------------------------------------------
    # Extract parsed Pydantic result
    # --------------------------------------------------------

    parsed = (
        completion
        .choices[0]
        .message
        .parsed
    )


    if parsed is None:

        raise RuntimeError(
            "Gemini returned no structured output."
        )


    # --------------------------------------------------------
    # Convert Pydantic object into dictionary
    # --------------------------------------------------------

    return parsed.model_dump()