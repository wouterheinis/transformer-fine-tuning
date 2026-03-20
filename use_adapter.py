from __future__ import annotations

import argparse
from typing import List

import chess
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = "You are a strong chess engine.\nFEN: {fen}\nMOVE:"


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LoraChessScorer:
    def __init__(self, base_model_name: str, adapter_path: str):
        self.dev = device()
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.to(self.dev)
        self.model.eval()

    def _encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"].to(self.dev)

    @torch.inference_mode()
    def score_move(self, fen: str, move_uci: str) -> float:
        prompt = PROMPT_TEMPLATE.format(fen=fen)
        full_text = prompt + " " + move_uci

        input_ids = self._encode(full_text)
        prompt_ids = self._encode(prompt)

        labels = input_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100

        out = self.model(input_ids=input_ids, labels=labels)

        # loss is average negative log-likelihood on the non-masked tokens
        return -float(out.loss.item())

    @torch.inference_mode()
    def rank_legal_moves(self, fen: str) -> List[tuple[str, float]]:
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        scores = [(mv, self.score_move(fen, mv)) for mv in legal_moves]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="distilgpt2")
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--fen", type=str, required=True)
    args = parser.parse_args()

    scorer = LoraChessScorer(args.base_model, args.adapter)
    ranked = scorer.rank_legal_moves(args.fen)

    print("Top 10 moves:")
    for mv, sc in ranked[:10]:
        print(f"{mv:8s} {sc:.4f}")


if __name__ == "__main__":
    main()
