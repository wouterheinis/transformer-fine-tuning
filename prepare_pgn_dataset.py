from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import chess
import chess.pgn
from tqdm import tqdm


PROMPT_TEMPLATE = "You are a strong chess engine.\nFEN: {fen}\nMOVE:"
TARGET_TEMPLATE = " {move}"


def iter_games(pgn_path: Path) -> Iterable[chess.pgn.Game]:
    with pgn_path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game


def game_is_usable(game: chess.pgn.Game, min_elo: int) -> bool:
    white_elo = game.headers.get("WhiteElo")
    black_elo = game.headers.get("BlackElo")

    try:
        if white_elo is not None and int(white_elo) < min_elo:
            return False
        if black_elo is not None and int(black_elo) < min_elo:
            return False
    except ValueError:
        return False

    result = game.headers.get("Result", "")
    return result in {"1-0", "0-1", "1/2-1/2"}


def example_from_position(board: chess.Board, move: chess.Move) -> dict:
    fen = board.fen()
    uci = move.uci()
    prompt = PROMPT_TEMPLATE.format(fen=fen)
    completion = TARGET_TEMPLATE.format(move=uci)
    text = prompt + completion
    return {
        "fen": fen,
        "move": uci,
        "prompt": prompt,
        "completion": completion,
        "text": text,
    }


def write_jsonl(examples: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--valid-out", type=Path, required=True)
    parser.add_argument("--max-games", type=int, default=5000)
    parser.add_argument("--min-elo", type=int, default=1800)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--max-plies-per-game", type=int, default=120)
    args = parser.parse_args()

    examples: list[dict] = []
    kept_games = 0

    for game in tqdm(iter_games(args.pgn), desc="Reading PGN"):
        if kept_games >= args.max_games:
            break
        if not game_is_usable(game, min_elo=args.min_elo):
            continue

        board = game.board()
        ply = 0

        for move in game.mainline_moves():
            if ply >= args.max_plies_per_game:
                break

            if move in board.legal_moves:
                examples.append(example_from_position(board, move))
                board.push(move)
                ply += 1
            else:
                break

        kept_games += 1

    if not examples:
        raise RuntimeError("No training examples were produced. Check your PGN file and filters.")

    split_idx = int(len(examples) * (1.0 - args.valid_ratio))
    train_examples = examples[:split_idx]
    valid_examples = examples[split_idx:]

    write_jsonl(train_examples, args.train_out)
    write_jsonl(valid_examples, args.valid_out)

    print(f"Wrote {len(train_examples)} train examples to {args.train_out}")
    print(f"Wrote {len(valid_examples)} valid examples to {args.valid_out}")


if __name__ == "__main__":
    main()
