from __future__ import annotations

import argparse
import json
import os
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

from src import utils
from src.qa_prompt_builder import (
    ChinesePathGenerationWithAnswerPromptBuilder,
    PathGenerationWithAnswerPromptBuilder,
)
from src.utils.qa_utils import eval_pure_path_result
from vllm_gcr.model import VLLMGraphConstrainedModel


def merge_rule_result(qa_dataset, rule_dataset, n_proc: int = 1, filter_empty: bool = False):
    question_to_rule = {}
    for data in rule_dataset:
        qid = data["id"]
        question_to_rule[qid] = {
            "predicted_paths": data["prediction"],
            "ground_paths": data["ground_paths"],
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0,
            num_proc=n_proc,
        )
    return qa_dataset


def get_output_file(path: str, force: bool = False):
    if not os.path.exists(path) or force:
        fout = open(path, "w", encoding="utf-8")
        return fout, []

    processed_results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                result = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Error in line: {line}") from exc
            processed_results.append(result["id"])
    fout = open(path, "a", encoding="utf-8")
    return fout, processed_results


def get_input_builder(args, tokenizer):
    if args.prompt_language == "zh":
        if args.prompt_mode == "few-shot":
            raise ValueError("Chinese prompt builder does not implement few-shot mode.")
        return ChinesePathGenerationWithAnswerPromptBuilder(
            tokenizer,
            args.prompt_mode,
            index_path_length=args.index_path_length,
            undirected=args.undirected,
            add_rule=args.add_rule,
        )
    return PathGenerationWithAnswerPromptBuilder(
        tokenizer,
        args.prompt_mode,
        index_path_length=args.index_path_length,
        undirected=args.undirected,
        add_rule=args.add_rule,
    )


def build_prediction_request(
    sample: dict[str, Any],
    input_builder,
    model: VLLMGraphConstrainedModel,
    enable_constrained_by_default: bool,
):
    input_query, ground_paths, trie = input_builder.process_input(sample)
    if trie is None:
        return None

    start_token_id = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_START_TOKEN)
    end_token_id = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_END_TOKEN)
    llm_input = model.prepare_model_prompt(input_query)
    request = model.build_request(
        prompt=llm_input,
        trie=trie,
        start_token_id=start_token_id,
        end_token_id=end_token_id,
        enable_constrained_by_default=enable_constrained_by_default,
    )
    return {
        "id": sample["id"],
        "question": sample["question"],
        "ground_truth": sample["answer"],
        "ground_truth_paths": ground_paths,
        "input": llm_input,
        "request": request,
    }


def write_batch_predictions(
    batch_items: list[dict[str, Any]],
    model: VLLMGraphConstrainedModel,
    fout,
    debug: bool = False,
) -> None:
    if not batch_items:
        return

    requests = [item["request"] for item in batch_items]
    try:
        predictions = model.generate_batch(requests)
    except Exception as exc:
        print(
            f"Batch generation failed for {len(batch_items)} request(s): {exc}. "
            "Falling back to single-request generation."
        )
        predictions = []
        for item in batch_items:
            try:
                predictions.append(model.generate_batch([item["request"]])[0])
            except Exception as inner_exc:
                print(f"Generation failed for sample {item['id']}: {inner_exc}")
                predictions.append(None)

    for item, prediction in zip(batch_items, predictions):
        if prediction is None:
            print("None result for: ", item["id"])
            continue

        result = {
            "id": item["id"],
            "question": item["question"],
            "prediction": prediction,
            "ground_truth": item["ground_truth"],
            "ground_truth_paths": item["ground_truth_paths"],
            "input": item["input"],
        }
        if debug:
            print(json.dumps(result, ensure_ascii=False))
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        fout.flush()


def main(args) -> None:
    input_file = os.path.join(args.data_path, args.d)
    dataset = load_dataset(input_file, split=args.split)

    post_fix = (
        f"{args.prefix}{args.prompt_mode}-{args.generation_mode}-"
        f"k{args.k}-index_len{args.index_path_length}"
    )
    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = utils.load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
        post_fix += "_" + rule_postfix

    data_name = args.d + "_undirected" if args.undirected else args.d
    output_dir = os.path.join(
        args.predict_path,
        data_name,
        args.model_name,
        args.split,
        post_fix,
    )
    print("Save results to: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model = VLLMGraphConstrainedModel(args)
    print("Prepare pipeline for inference...")
    model.prepare_for_inference()
    input_builder = get_input_builder(args, model.tokenizer)

    with open(os.path.join(output_dir, "args.txt"), "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=2, ensure_ascii=False)

    output_path = os.path.join(output_dir, "predictions.jsonl")
    fout, processed_list = get_output_file(output_path, force=args.force)

    batch_items: list[dict[str, Any]] = []
    processed_set = set(processed_list)
    try:
        for sample in tqdm(dataset, total=len(dataset)):
            if sample["id"] in processed_set:
                continue

            item = build_prediction_request(
                sample=sample,
                input_builder=input_builder,
                model=model,
                enable_constrained_by_default=args.enable_constrained_by_default,
            )
            if item is None:
                print("None result for: ", sample["id"])
                continue

            batch_items.append(item)
            if len(batch_items) >= args.batch_size:
                write_batch_predictions(batch_items, model, fout, debug=args.debug)
                batch_items = []

        write_batch_predictions(batch_items, model, fout, debug=args.debug)
    finally:
        fout.close()

    if not args.disable_eval:
        eval_pure_path_result(output_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, default="rmanluo")
    argparser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    argparser.add_argument("--split", type=str, default="test[:100]")
    argparser.add_argument("--index_path_length", type=int, default=2)
    argparser.add_argument("--predict_path", type=str, default="results/GenPaths")
    argparser.add_argument("--force", action="store_true", help="Force overwrite results.")
    argparser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Only used by dataset.map/filter when merging rule files.",
    )
    argparser.add_argument(
        "--undirected",
        type=lambda x: str(x).lower() == "true",
        default=False,
    )
    argparser.add_argument("--debug", action="store_true", help="Print debug information.")
    argparser.add_argument(
        "--prompt_mode",
        type=str,
        default="zero-shot",
        choices=["zero-shot", "mcq-zero-shot", "few-shot"],
    )
    argparser.add_argument("--prompt_language", choices=["zh", "en"], default="zh")
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--add_rule", action="store_true")
    argparser.add_argument("--disable_eval", action="store_true")
    argparser.add_argument(
        "--enable_constrained_by_default",
        action="store_true",
        help="Enable constraints from the first generated token even without <PATH>.",
    )
    argparser.add_argument(
        "--rule_path",
        type=str,
        default=(
            "results/gen_rule_path/webqsp_undirected/"
            "Llama-2-7b-chat-hf_align-spectoken-joint/test/predictions_3_False.jsonl"
        ),
    )
    argparser.add_argument("--prefix", type=str, default="")

    VLLMGraphConstrainedModel.add_args(argparser)
    args = argparser.parse_args()
    main(args)
