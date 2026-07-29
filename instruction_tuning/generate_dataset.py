import csv
import json
import requests
import sys
import time

API_URL = "http://127.0.0.1:11434/v1/chat/completions"

PROMPT_TEMPLATE = """
You are creating an instruction tuning dataset.

Given the story below, generate exactly 3 different user instructions that all point to the story as the answer.
Instruction styles could be:
- Generate a short story
- Write a children's story
- Tell me a bedtime story
- Create a story about ...
- Write a moral story
- Write a funny story
- Continue the story
- Write a story with a happy ending
- Tell a story for kids
- Write an educational story

Return ONLY a valid JSON array of objects, where each object has an "instruction" and a "response". The "response" MUST be exactly the original story for all items. Do not output anything else.

Format:
[
  {{
    "instruction": "...",
    "response": "..."
  }},
  ...
]

Story:
{story}
"""

def main():
    output = []
    max_stories = 1000
    
    print(f"Reading up to {max_stories} stories from validation.csv...")
    
    with open("../dataset/validation.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        
        count = 0
        for row in reader:
            if not row:
                continue
                
            story = row[0]
            if len(story.strip()) < 50:
                continue

            prompt = PROMPT_TEMPLATE.format(story=story)

            try:
                response = requests.post(
                    API_URL,
                    json={
                        "model": "phi3",
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.7
                    },
                    timeout=120
                )
                response.raise_for_status()
                
                text = response.json()["choices"][0]["message"]["content"]
                
                text = text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

                arr = json.loads(text)
                if isinstance(arr, list):
                    for item in arr:
                        if "instruction" in item and "response" in item:
                            item["response"] = story
                            output.append(item)
            except Exception as e:
                print(f"Error on story {count}: {e}")
                print(f"Raw output: {text}")
                
            count += 1
            if count % 10 == 0:
                print(f"Processed {count}/{max_stories} stories. Generated {len(output)} pairs so far.")
            
            if count >= max_stories:
                break

    with open("instruction_dataset.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(output)} instruction examples to instruction_dataset.json")

if __name__ == "__main__":
    main()
