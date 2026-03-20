# Transformer Chess Fine-Tuning with LoRA

This repo fine-tunes a DistilGPT-2 chess move scorer with LoRA.

## What it does

Given a chess position in FEN format, the model learns to predict the next move in UCI format.

Training example format:

You are a strong chess engine.
FEN: <fen>
MOVE: <uci_move>

The resulting LoRA adapter can be plugged into a larger chess bot that already uses legal move filtering, heuristics, and search.

## Install

```bash
pip install -r requirements.txt
