# benchmark_inference.py

import torch
import os
import sys
import time

from gpt import tinyStory

# Tokenizer import
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'Tokenizer', 'v2')
    )
)

from tokenizer import Tokenizer


# -----------------------------
# Configuration
# -----------------------------

BLOCK_SIZE = 256
VOCAB_SIZE = 4097

EMBEDDING_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 4

DROPOUT = 0.0

WEIGHTS_PATH = "StoryGenerater.pth"

MAX_NEW_TOKENS = 80


# -----------------------------
# Test Prompts
# -----------------------------

TEST_PROMPTS = [

    # TinyStories style
    "Once upon a time",
    "One sunny morning",
    "There was a little girl named Lily",
    "There was a boy named Timmy",
    "In a big green forest",

    # Animals
    "A little rabbit",
    "A friendly dog",
    "A small bird",
    "A clever fox",
    "A happy cat",

    # Story continuation
    "One day",
    "The little boy",
    "The little girl",
    "Mom said",
    "Dad smiled",

    # Friendship
    "Tom and Lily",
    "Ben found",
    "Two best friends",
    "The children wanted",
    "Everyone was happy",

    # Unseen concepts
    "The robot woke up",
    "An astronaut landed",
    "A pirate found",
    "The dragon smiled",

    # Long context
    (
        "Once upon a time there was a little rabbit "
        "who lived near a beautiful forest with all "
        "of his friends"
    ),

    # Random prompts
    "I want to do something",
    "Hello",
]


TEMPERATURES = [
    0.2,
    0.5,
    0.8,
    1.0,
    1.2
]


# -----------------------------
# Load Model
# -----------------------------

def load_model(device):

    print("Loading tokenizer...")
    tokenizer = Tokenizer()

    print("Loading model...")

    model = tinyStory(
        vocab_size=VOCAB_SIZE,
        context_size=BLOCK_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )


    if not os.path.exists(WEIGHTS_PATH):
        print(
            f"Missing weights file: {WEIGHTS_PATH}"
        )
        sys.exit(1)


    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location=device,
        weights_only=True
    )


    model.load_state_dict(checkpoint)

    model.to(device)

    model.eval()

    return model, tokenizer



# -----------------------------
# Generate
# -----------------------------

def generate_story(
        model,
        tokenizer,
        prompt,
        device,
        temperature=0.8,
        top_k=40
):

    formatted_prompt = f"User: {prompt}\nModel: "
    tokens = tokenizer.encode(prompt)


    x = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )


    start = time.perf_counter()


    with torch.no_grad():

        output = model.generate(
            x,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_token_id
        )


    end = time.perf_counter()


    generated_tokens = output[0].tolist()

    text = tokenizer.decode(
        generated_tokens
    )
    
    # Parse response
    if "Model: " in text:
        story = text.split("Model: ")[-1]
    else:
        story = text
        
    story = story.replace("<|endoftext|>", "").strip()


    latency = (end-start)*1000


    return story, latency



# -----------------------------
# Run Benchmark
# -----------------------------
device = (
        "cuda"
        if torch.cuda.is_available()
        else
        "mps"
        if torch.backends.mps.is_available()
        else
        "cpu"
    )

print(
        f"\nUsing device: {device}\n"
    )


model, tokenizer = load_model(device)

def run_benchmark():
    print("\n==============================")
    print(" TinyStory GPT Benchmark")
    print("==============================\n")


    total_time = 0
    count = 0


    for i, prompt in enumerate(TEST_PROMPTS):

        print("="*80)

        print(
            f"TEST {i+1}/{len(TEST_PROMPTS)}"
        )

        print(
            "PROMPT:",
            repr(prompt)
        )

        print("-"*80)


        output, latency = generate_story(
            model,
            tokenizer,
            prompt,
            device,
            temperature=0.8
        )


        print(output)

        print()

        print(
            f"Inference time: {latency:.2f} ms"
        )


        total_time += latency
        count += 1



    print("\n\n==============================")
    print(" Temperature Test")
    print("==============================\n")


    prompt = "Once upon a time"


    for temp in TEMPERATURES:

        print("="*80)

        print(
            f"Temperature: {temp}"
        )

        output, latency = generate_story(
            model,
            tokenizer,
            prompt,
            device,
            temperature=temp
        )


        print(output)

        print(
            f"\nTime: {latency:.2f} ms"
        )



    print("\n==============================")

    print(
        f"Average inference time: "
        f"{total_time/count:.2f} ms"
    )

    print("==============================")


def run_inference(): 
    prompts=['Write a story about a blue elephant.',
              'Tell me a story about two robots.',
              'Write two bedtime stories.',
              'Tell a funny story.',
              'Write a story in three paragraphs.',
              'Write a story whose moral is honesty.',
              'Write a story about space.',
              'Write a short story about a dinosaur.'
              'Write two short stories.'
              'Tell a story about a robot.'
              ,'once upon a time'
              ]
    prompt=input("enter your prompt : ")
    prompts=[prompt]
    for prompt in prompts:
        print("="*80)
        print(
            f"PROMPT: {prompt}"
        )
        print("response: ")  
        output, latency = generate_story(
                model,
                tokenizer,
                prompt,
                device,
                temperature=0.8
            )
        print(output)
        print()
    print("Latency Time : "+str(latency))
if __name__ == "__main__":
    #print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} Million")
    run_inference()