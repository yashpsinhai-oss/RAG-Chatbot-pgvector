import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import generation
import search

EXPERIMENT_VERSION="v2"
EXTRACTION_PROMPT_VERSION="v1"
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

    16. A citation appearing at the end of a sentence applies
    to all factual claims extracted from that sentence,
    including claims created by splitting it.

    17. Do not inherit citations from separate sentences,
    headings, or bullet points.

    18. For an introductory statement followed by bullet points,
    do not automatically inherit citations from the bullet points.
    Only use citations explicitly associated with the
    introductory statement.

    CONTEXT:{answer}
    """

    #changed the prompt by adding 16 17 18 for the citation extraction issue
    #but it still appears in our generated citation and claim extraction  follows our citation association rule to not pick the citation number from the bullet pts or neighbour
    #seems like the issue of prompt used for generated answer citation claim association in generation

    response=client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role":"user","content":prompt}],
        response_format={"type": "json_object"}
    )

    #also return the prompt variable as we did in generation.py to get hold of our prompt along with the dynamic text answer
    return prompt,response

def get_claims(answer):
    prompt,response=extract_claims(answer)
    claims_data=json.loads(response.choices[0].message.content)
    return claims_data


if __name__=="__main__":
    #to generate run specific folders at every run
    experiment_dir=Path("evaluation")/EXPERIMENT_VERSION
    experiment_dir.mkdir(parents=True,exist_ok=True)

    run_num=1
    while(experiment_dir/f"run_{run_num:03d}").exists():
        run_num+=1

    run_id = f"run_{run_num:03d}"
    run_dir = experiment_dir / run_id
    run_dir.mkdir()

    extraction_dir=run_dir/"extraction"
    extraction_dir.mkdir()    

    query=input("Ask your question: ")

    #saving the metadata for every run, prompt and experiment version,query
    run_metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "run_id": run_id,
        "query": query,
        "extraction_prompt_version": EXTRACTION_PROMPT_VERSION
    }
    with open(run_dir/ "run_metadata.json", "w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=4)

    result=search.search_chunks(query)
    if not result:
        print("More context is needed.")
        sys.exit()
    join_attribution_text=generation.prepare_attribution(result)  
    print(join_attribution_text)  

    prompt,answer=generation.generate_answer(query,join_attribution_text)
    # Save generated answer
    with open(run_dir/"generated_answer.txt", "w", encoding="utf-8") as file:
        file.write(answer)
    #saving extraction prompt also along with the claims and citations
    with open(extraction_dir/f"extraction_prompt_{EXTRACTION_PROMPT_VERSION}.txt","w",encoding="utf-8") as file:
        file.write(prompt)
    print(answer)
    claims_data=get_claims(answer)
    #save generated claims
    with open(run_dir/"extracted_claims.json", "w", encoding="utf-8") as file:
        json.dump(claims_data, file, indent=4)
    print(json.dumps(claims_data,indent=4))
    