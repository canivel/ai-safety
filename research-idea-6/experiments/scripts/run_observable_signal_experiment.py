#!/usr/bin/env python
"""
Run the observable-signal experiment by generating stereo completions, scoring them,
and producing comparison summaries for Base vs Chat models.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.model_diffing import StereoModelRunner
from src.observable_signals import score_prompt_set

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = BASE_DIR / "data" / "observable_signal_minimal_pairs.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stereo inference on observable-signal prompts and score outputs."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to observable-signal minimal pairs JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saving responses, scores, and summaries.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HuggingFace token for Gemma access.",
    )
    parser.add_argument(
        "--backend",
        choices=("transformers", "transformer_lens"),
        default="transformers",
        help="Backend for StereoModelRunner (default: transformers).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Computation device (default: {DEFAULT_DEVICE}).",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_MAP.keys()),
        default="float16",
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum new tokens to generate per completion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (used when --do-sample).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling threshold (used when --do-sample with transformers backend).",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use stochastic sampling instead of greedy decoding (transformers backend).",
    )
    parser.add_argument(
        "--no-stop-at-eos",
        action="store_true",
        help="Allow generation to continue past EOS tokens.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: List[float]) -> float:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return 0.0
    return float(sum(filtered) / len(filtered))


def compute_prompt_summary(
    model_name: str, item_id: str, domain: str, scores: Dict
) -> Dict[str, object]:
    variant_scores = {entry["variant"]: entry for entry in scores["variants"]}
    pairwise = scores["pairwise"]

    instability_rate = mean([1.0 if entry["polarity_shift"] else 0.0 for entry in pairwise])
    avg_similarity = mean([entry["similarity_to_neutral"] for entry in pairwise]) if pairwise else 1.0

    assertive_variants = [variant_scores.get(name) for name in ("confident", "authority")]
    deferential_variants = [variant_scores.get(name) for name in ("uncertain", "emotional")]

    agreement_assertive = mean(
        [entry["agreement_count"] for entry in assertive_variants if entry is not None]
    )
    agreement_deferential = mean(
        [entry["agreement_count"] for entry in deferential_variants if entry is not None]
    )
    agreement_gradient = agreement_assertive - agreement_deferential

    confidence_gap = mean(
        [entry["confidence_evidence_gap"] for entry in variant_scores.values()]
    )
    counterfactual_resistance = mean(
        [entry["counterfactual_resistance"] for entry in variant_scores.values()]
    )

    return {
        "model": model_name,
        "id": item_id,
        "domain": domain,
        "instability_rate": instability_rate,
        "avg_similarity_to_neutral": avg_similarity,
        "agreement_gradient": agreement_gradient,
        "confidence_evidence_gap": confidence_gap,
        "counterfactual_resistance": counterfactual_resistance,
    }


def aggregate_by_domain(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    domain_rows: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        domain_rows.setdefault(row["domain"], []).append(row)

    aggregates: List[Dict[str, object]] = []
    for domain, items in sorted(domain_rows.items()):
        aggregates.append(
            {
                "model": items[0]["model"],
                "domain": domain,
                "num_prompts": len(items),
                "avg_instability_rate": mean([item["instability_rate"] for item in items]),
                "avg_similarity_to_neutral": mean(
                    [item["avg_similarity_to_neutral"] for item in items]
                ),
                "avg_agreement_gradient": mean(
                    [item["agreement_gradient"] for item in items]
                ),
                "avg_confidence_evidence_gap": mean(
                    [item["confidence_evidence_gap"] for item in items]
                ),
                "avg_counterfactual_resistance": mean(
                    [item["counterfactual_resistance"] for item in items]
                ),
            }
        )

    if rows:
        aggregates.append(
            {
                "model": rows[0]["model"],
                "domain": "ALL",
                "num_prompts": len(rows),
                "avg_instability_rate": mean([item["instability_rate"] for item in rows]),
                "avg_similarity_to_neutral": mean(
                    [item["avg_similarity_to_neutral"] for item in rows]
                ),
                "avg_agreement_gradient": mean(
                    [item["agreement_gradient"] for item in rows]
                ),
                "avg_confidence_evidence_gap": mean(
                    [item["confidence_evidence_gap"] for item in rows]
                ),
                "avg_counterfactual_resistance": mean(
                    [item["counterfactual_resistance"] for item in rows]
                ),
            }
        )

    return aggregates


def run_generation(
    runner: StereoModelRunner,
    dataset: List[Dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    stop_at_eos: bool,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    raw_outputs = []
    base_payload = []
    chat_payload = []

    for idx, item in enumerate(dataset, start=1):
        prompt_id = item["id"]
        domain = item.get("domain", "unknown")
        print(f"[{idx}/{len(dataset)}] Generating responses for {prompt_id} ({domain})...")

        base_variants: Dict[str, str] = {}
        chat_variants: Dict[str, str] = {}
        for variant_name, prompt in item["variants"].items():
            base_text, chat_text = runner.generate_completions(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                stop_at_eos=stop_at_eos,
            )
            base_variants[variant_name] = base_text
            chat_variants[variant_name] = chat_text

        raw_outputs.append(
            {
                "id": prompt_id,
                "domain": domain,
                "question": item.get("question"),
                "expected_conclusion": item.get("expected_conclusion"),
                "responses": {
                    "base": base_variants,
                    "chat": chat_variants,
                },
            }
        )

        base_payload.append({"id": prompt_id, "domain": domain, "responses": base_variants})
        chat_payload.append({"id": prompt_id, "domain": domain, "responses": chat_variants})

    return raw_outputs, base_payload, chat_payload


def score_model_payload(
    model_name: str, payload: List[Dict[str, object]]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    scored = []
    summaries = []
    for item in payload:
        result = score_prompt_set(item["responses"])
        scored.append({"id": item["id"], "domain": item["domain"], "scores": result})
        summaries.append(
            compute_prompt_summary(model_name, item["id"], item["domain"], result)
        )
    return scored, summaries


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "domain",
        "num_prompts",
        "avg_instability_rate",
        "avg_similarity_to_neutral",
        "avg_agreement_gradient",
        "avg_confidence_evidence_gap",
        "avg_counterfactual_resistance",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)

    dtype = DTYPE_MAP[args.dtype]
    use_transformer_lens = args.backend == "transformer_lens"
    runner = StereoModelRunner(
        device=args.device,
        use_transformer_lens=use_transformer_lens,
        dtype=dtype,
    )

    print("Loading models...")
    runner.load_models(hf_token=args.hf_token)
    print("Models ready. Running observable-signal prompts.")

    stop_at_eos = not args.no_stop_at_eos
    raw_outputs, base_payload, chat_payload = run_generation(
        runner,
        dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        stop_at_eos=stop_at_eos,
    )

    base_scores, base_summaries = score_model_payload("base", base_payload)
    chat_scores, chat_summaries = score_model_payload("chat", chat_payload)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    responses_path = output_dir / "observable_signals_responses.json"
    scores_path = output_dir / "observable_signals_scores.json"
    summary_csv_path = output_dir / "observable_signals_summary.csv"

    with responses_path.open("w", encoding="utf-8") as handle:
        json.dump(raw_outputs, handle, indent=2)

    scores_payload = {
        "base": {"scores": base_scores, "prompt_summary": base_summaries},
        "chat": {"scores": chat_scores, "prompt_summary": chat_summaries},
    }
    with scores_path.open("w", encoding="utf-8") as handle:
        json.dump(scores_payload, handle, indent=2)

    summary_rows = aggregate_by_domain(base_summaries) + aggregate_by_domain(chat_summaries)
    write_summary_csv(summary_csv_path, summary_rows)

    print(f"\nSaved raw responses to: {responses_path}")
    print(f"Saved per-prompt scores to: {scores_path}")
    print(f"Saved aggregate summary to: {summary_csv_path}")


if __name__ == "__main__":
    main()
