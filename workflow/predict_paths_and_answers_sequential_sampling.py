import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

from datasets import load_dataset
from tqdm import tqdm
from transformers import StoppingCriteriaList

from src import utils
from src.graph_constrained_decoding import GraphConstrainedDecoding, PathEndStoppingCriteria
from src.llms import get_registed_model
from src.qa_prompt_builder import ChinesePathGenerationWithAnswerPromptBuilder
from src.trie import Trie
from src.utils.qa_utils import eval_pure_path_result


def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w", encoding="utf-8")
        return fout, []
    else:
        with open(path, "r", encoding="utf-8") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except Exception as exc:
                    raise ValueError(f"Error in line: {line}") from exc
                processed_results.append(results["id"])
        fout = open(path, "a", encoding="utf-8")
        return fout, processed_results


def normalize_token_sequence(sequence):
    return tuple(int(token) for token in sequence)


def canonicalize_generated_path(path_text):
    return " ".join(str(path_text).split()).strip()


def build_filtered_trie(trie, blocked_paths):
    if not blocked_paths:
        return trie

    remaining_sequences = [
        list(sequence)
        for sequence in trie
        if normalize_token_sequence(sequence) not in blocked_paths
    ]
    if not remaining_sequences:
        return None
    return Trie(remaining_sequences)


def get_sequence_length(sequence, prompt_len, eos_token_id):
    if eos_token_id is None:
        return sequence.shape[0]
    generated_tokens = sequence[prompt_len:]
    eos_positions = (generated_tokens == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) == 0:
        return sequence.shape[0]
    return prompt_len + eos_positions[0].item() + 1


def extract_generated_path_tokens(sequence, sequence_length, start_token_id, end_token_id):
    trimmed_sequence = sequence[:sequence_length].tolist()
    try:
        start_index = max(
            index for index, token_id in enumerate(trimmed_sequence) if token_id == start_token_id
        )
    except ValueError:
        return None

    path_tokens = trimmed_sequence[start_index:]
    if end_token_id in path_tokens:
        end_index = path_tokens.index(end_token_id) + 1
        path_tokens = path_tokens[:end_index]

    return tuple(path_tokens) if path_tokens else None


def build_generation_kwargs(model, gcr, stopping_criteria, num_generations):
    generation_kwargs = {
        "max_new_tokens": model.generation_cfg.max_new_tokens,
        "num_return_sequences": num_generations,
        "prefix_allowed_tokens_fn": gcr.allowed_tokens_fn,
        "stopping_criteria": stopping_criteria,
        "pad_token_id": model.tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "do_sample": True,
    }

    temperature = getattr(model.generation_cfg, "temperature", None)
    if temperature is not None:
        generation_kwargs["temperature"] = temperature

    top_p = getattr(model.generation_cfg, "top_p", None)
    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    top_k = getattr(model.generation_cfg, "top_k", None)
    if top_k is not None and top_k > 0:
        generation_kwargs["top_k"] = top_k

    return generation_kwargs


def generate_paths_once(model, llm_input, trie, num_generations, start_token_id, end_token_id):
    inputs = model.tokenizer(llm_input, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(model.model.device)
    attention_mask = inputs.attention_mask.to(model.model.device)
    gcr = GraphConstrainedDecoding(
        model.tokenizer,
        trie,
        start_token_id,
        end_token_id,
        True,
    )
    stopping_criteria = StoppingCriteriaList(
        [PathEndStoppingCriteria(start_token_id, end_token_id)]
    )
    generation_kwargs = build_generation_kwargs(model, gcr, stopping_criteria, num_generations)

    outputs = model.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_kwargs,
    )

    prompt_len = input_ids.shape[1]
    generated_texts = [
        model.tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True)
        for sequence in outputs.sequences
    ]
    sequence_lengths = [
        get_sequence_length(sequence, prompt_len, model.tokenizer.eos_token_id)
        for sequence in outputs.sequences
    ]
    return generated_texts, outputs.sequences, sequence_lengths

def generate_k_paths_with_grpo_resampling(
    model,
    llm_input,
    trie,
    start_token_id,
    end_token_id,
    k,
    max_resample_rounds,
):
    unique_texts = []
    seen_paths = set()
    blocked_path_tokens = set()
    stagnant_rounds = 0

    for _ in range(max_resample_rounds):
        remaining = k - len(unique_texts)
        if remaining <= 0:
            break

        filtered_trie = build_filtered_trie(trie, blocked_path_tokens)
        if filtered_trie is None:
            break

        round_texts, round_sequences, round_lengths = generate_paths_once(
            model,
            llm_input,
            filtered_trie,
            remaining,
            start_token_id,
            end_token_id,
        )
        new_unique_count = 0

        for text, sequence, length in zip(round_texts, round_sequences, round_lengths):
            generated_path_tokens = extract_generated_path_tokens(
                sequence,
                length,
                start_token_id,
                end_token_id,
            )
            if generated_path_tokens is not None and generated_path_tokens not in blocked_path_tokens:
                blocked_path_tokens.add(generated_path_tokens)

            canonical_path = canonicalize_generated_path(text)
            if not canonical_path or canonical_path in seen_paths:
                continue

            seen_paths.add(canonical_path)
            unique_texts.append(text)
            new_unique_count += 1

            if len(unique_texts) >= k:
                break

        stagnant_rounds = stagnant_rounds + 1 if new_unique_count == 0 else 0
        if stagnant_rounds >= 2:
            break

    if len(unique_texts) == 0:
        return None
    if k == 1 and len(unique_texts) == 1:
        return unique_texts[0]
    return unique_texts


def prediction(data, processed_list, input_builder, model, k, max_resample_rounds):
    question = data["question"]
    answer = data["answer"]
    sample_id = data["id"]
    if sample_id in processed_list:
        return None

    input_query, ground_paths, trie = input_builder.process_input(data)
    if trie is None:
        return None

    start_token_id = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_START_TOKEN)
    end_token_id = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_END_TOKEN)
    llm_input = model.prepare_model_prompt(input_query)
    prediction_result = generate_k_paths_with_grpo_resampling(
        model,
        llm_input,
        trie,
        start_token_id,
        end_token_id,
        k,
        max_resample_rounds,
    )
    if prediction_result is None:
        return None

    result = {
        "id": sample_id,
        "question": question,
        "prediction": prediction_result,
        "ground_truth": answer,
        "ground_truth_paths": ground_paths,
        "input": llm_input,
    }
    return result


def main(args, LLM):
    if args.generation_mode != "sampling":
        raise ValueError("该脚本仅支持 generation_mode=sampling")

    input_file = os.path.join(args.data_path, args.d)
    dataset = load_dataset(input_file, split=args.split)
    post_fix = (
        f"{args.prefix}{args.prompt_mode}-sampling-sequential-k{args.k}-index_len{args.index_path_length}"
    )
    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = utils.load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
        post_fix += "_" + rule_postfix
    data_name = args.d + "_undirected" if args.undirected else args.d
    output_dir = os.path.join(args.predict_path, data_name, args.model_name, args.split, post_fix)
    print("Save results to: ", output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = LLM(args)
    print("Prepare pipeline for inference...")
    model.prepare_for_inference()
    input_builder = ChinesePathGenerationWithAnswerPromptBuilder(
        model.tokenizer,
        args.prompt_mode,
        index_path_length=args.index_path_length,
        undirected=args.undirected,
        add_rule=args.add_rule,
    )

    with open(os.path.join(output_dir, "args.txt"), "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=2, ensure_ascii=False)

    fout, processed_list = get_output_file(
        os.path.join(output_dir, "predictions.jsonl"),
        force=args.force,
    )

    if args.n > 1:
        with Pool(args.n) as p:
            for res in tqdm(
                p.imap(
                    partial(
                        prediction,
                        processed_list=processed_list,
                        input_builder=input_builder,
                        model=model,
                        k=args.k,
                        max_resample_rounds=args.max_resample_rounds,
                    ),
                    dataset,
                ),
                total=len(dataset),
            ):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res, ensure_ascii=False))
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                    fout.flush()
    else:
        for data in tqdm(dataset):
            res = prediction(
                data,
                processed_list,
                input_builder,
                model,
                args.k,
                args.max_resample_rounds,
            )
            if res is not None:
                if args.debug:
                    print(json.dumps(res, ensure_ascii=False))
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
            else:
                print("None result for: ", data["id"])
    fout.close()

    eval_pure_path_result(os.path.join(output_dir, "predictions.jsonl"))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, default="rmanluo")
    argparser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    argparser.add_argument("--split", type=str, default="test[:100]")
    argparser.add_argument("--index_path_length", type=int, default=2)
    argparser.add_argument("--predict_path", type=str, default="results/GenPaths")
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="gcr-Llama-2-7b-chat-hf",
    )
    argparser.add_argument("--force", action="store_true", help="force to overwrite the results")
    argparser.add_argument("--n", type=int, default=1, help="number of processes")
    argparser.add_argument("--undirected", type=lambda x: (str(x).lower() == "true"), default=False)
    argparser.add_argument("--debug", action="store_true", help="print debug information")
    argparser.add_argument(
        "--prompt_mode",
        type=str,
        default="zero-shot",
        choices=["zero-shot", "mcq-zero-shot", "few-shot"],
    )
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--add_rule", action="store_true")
    argparser.add_argument("--max_resample_rounds", type=int, default=4)
    argparser.add_argument(
        "--rule_path",
        type=str,
        default="results/gen_rule_path/webqsp_undirected/Llama-2-7b-chat-hf_align-spectoken-joint/test/predictions_3_False.jsonl",
    )
    argparser.add_argument("--prefix", type=str, default="")

    args, _ = argparser.parse_known_args()

    LLM = get_registed_model(args.model_name)
    LLM.add_args(argparser)

    args = argparser.parse_args()

    main(args, LLM)
