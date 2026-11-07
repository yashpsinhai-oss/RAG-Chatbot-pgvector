import os
import sys
import json
from dotenv import load_dotenv
from groq import Groq
import generation
import search

load_dotenv()
client=Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_claims(answer):
    prompt=f"""
    INSTRUCTIONS:
    1.Extract individual claims from the generated answer provided as context.
    2.Also extract the citation numbers corresponding to each claim from the answer provided as context
    3.Never invent or modify the citation numbers during extraction,preserve them exactly as it is
    4.Preserve the exact wording of each claim , dont invent or modify it
    5. If one sentence contains multiple independent factual claims,
    separate them into individual claims.
    6. Do not extract questions, opinions, or introductory statements.
    7. Return valid JSON only, following this structure:
    {{
    "claims": [
        {{
            "claim": "JWTs contain user information",
            "citations": [1]
        }},
        {{
            "claim": "JWTs can have an expiry date",
            "citations": [1]
        }}
    ]
    }}
    8. Extract claims only from statements explicitly present
   in the generated answer.

    9. Do not add technical details, explanations,
   examples or assumptions.

    10. Do not combine separate statements into a new claim.

    11. Preserve the original sentence wording exactly
    wherever possible.

    12. When a sentence contains multiple factual claims,
    split it into smaller claims using only words
    already present in the original sentence.

    13. Do not evaluate the correctness or evidence support
    of any claim. Your responsibility is extraction only.

    14. Associate citation numbers with the specific factual claim
   they support. Do not automatically assign citations from
   another sentence or section.

    15. If a sentence has no citation, return an empty citations list.

    CONTEXT:{answer}
    """
    response=client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role":"user","content":prompt}],
        response_format={"type": "json_object"}
    )
    return response

def get_claims(answer):
    response=extract_claims(answer)
    claims_data=json.loads(response.choices[0].message.content)
    return claims_data


if __name__=="__main__":
    query=input("Ask your question: ")
    result=search.search_chunks(query)
    if not result:
        print("More context is needed.")
        sys.exit()
    join_attribution_text=generation.prepare_attribution(result)  
    print(join_attribution_text)  
    answer=generation.generate_answer(query,join_attribution_text)
    print(answer)
    claims_data=get_claims(answer)
    print(json.dumps(claims_data,indent=4))
    