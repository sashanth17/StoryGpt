from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from inference import load_model, generate_story

app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
@app.get("/complete")
def complete_story(prompt:str):
    output, latency = generate_story(
                model,
                tokenizer,
                prompt,
                device,
                temperature=0.8
            )
    return {"output": output, "latency": latency}