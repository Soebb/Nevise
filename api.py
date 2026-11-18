import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hazm import Normalizer
from nevise import load_pre_model, spell_checking_on_sents


MODEL_DIR = os.getenv("NEVISE_MODEL_DIR", "model")
VOCAB_PATH = os.getenv("NEVISE_VOCAB_PATH", os.path.join(MODEL_DIR, "vocab.pkl"))
CHECKPOINT_PATH = os.getenv(
    "NEVISE_CHECKPOINT_PATH",
    os.path.join(MODEL_DIR, "model.pth.tar"),
)
TORCH_THREADS = int(os.getenv("NEVISE_TORCH_THREADS", "2"))


class CorrectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


class CorrectionResponse(BaseModel):
    corrected_text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Keep CPU resource usage predictable on shared PaaS instances.
    torch.set_num_threads(TORCH_THREADS)

    if not os.path.isfile(VOCAB_PATH):
        raise FileNotFoundError(f"Vocabulary file not found: {VOCAB_PATH}")
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    model, vocab, _ = load_pre_model(
        vocab_path=VOCAB_PATH,
        model_checkpoint_path=CHECKPOINT_PATH,
    )
    model.to("cpu")
    model.eval()

    app.state.model = model
    app.state.vocab = vocab
    app.state.device = "cpu"
    app.state.normalizer = Normalizer()
    yield


app = FastAPI(
    title="Nevise API",
    description="Persian spelling correction API powered by the Nevise model.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
    }


@app.post("/correct", response_model=CorrectionResponse)
def correct(request: CorrectionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    corrected_text = spell_checking_on_sents(
        model=app.state.model,
        vocab=app.state.vocab,
        device=app.state.device,
        normalizer=app.state.normalizer,
        txt=request.text,
    )
    return CorrectionResponse(corrected_text=corrected_text)
