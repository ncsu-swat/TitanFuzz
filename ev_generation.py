import argparse
import glob
import json
import multiprocessing
import os
import subprocess
import time

from model import LanguageModel  # Use the new modernized model class
from mycoverage import mp_executor
from process_file import clean_code, get_initial_programs
from util.clean_code import dead_code_elim
from util.instrumentor import SnippetInfill
from util.Logger import Logger
from util.Seed_pool import GA, GAR, GA_Coverage, GA_Random, GAR_depth
from util.util import ExecutionStatus, load_apis, run_cmd, set_seed
from validate import validate_status

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CURRENT_TIME = time.time()


def generate_loop(
    args, model: LanguageModel, original_codes: list, api: str, logger: Logger, max_valid: int
):
    num_selection = 1
    num_valid, generation_time, validation_time, total_run_time = (0, [], [], [])
    stats = {k: 0 for k in ["timeout", "exception", "crash", "duplicated", "notarget", "generated"]}
    total_outputs = set(original_codes)

    GA_class_map = {"random": GA_Random, "coverage": GA_Coverage, "fitness": GAR_depth}
    GA_class = GA_class_map.get(args.seed_selection_algo, GAR_depth)

    ga = GA_class(
        original_codes, num_selection, args.batch_size, args.folder, api,
        model.infill_ph, args.library, args.relaxargmut, args.seed_selection_algo,
        args.mutator_selection_algo, args.use_single_mutator, args.replace_type,
        args.seed_pool_size, args.mutator_set,
    )
    r = 0
    crashes = []
    total_programs = []
    while (max_valid < 0 or num_valid < max_valid) and sum(total_run_time) < args.timeout:
        logger.logo(f"--- Round : {r} ---")
        start_time_total = time.time()
        round_valid = 0
        selections = ga.selection()
        g_time, v_time = 0, 0
        for seed, infill_code, replace_type in selections:
            generations, filenames, add_flags = [], [], []

            start = time.time()
            outputs = model.generate(infill_code, num_samples=args.batch_size, do_sample=True)
            g_time += time.time() - start

            for output in outputs:
                output = clean_code(output, prints_and_imports=True, comment=True, cuda=True)
                output = dead_code_elim(output, api)
                stats["generated"] += 1
                if output in total_outputs:
                    stats["duplicated"] += 1
                    continue
                total_outputs.add(output)

                num_replaced, _, _ = SnippetInfill(
                    mask_identifier=model.infill_ph, api_call=api.split(".")[-1],
                    prefix=".".join(api.split(".")[1:-1]), library=args.library,
                    replace_type="argument",
                ).add_infill(output)

                start = time.time()
                status, msg = validate_status(
                    output, args.library, validate_mode=args.validate_mode,
                    test_executor=mp_executor.test_executor,
                )
                v_time += time.time() - start

                subfolder = ""
                dump_code = output

                if num_replaced < 1:
                    stats["notarget"] += 1
                    subfolder = "notarget"
                elif status == ExecutionStatus.SUCCESS:
                    subfolder = "valid"
                elif status == ExecutionStatus.TIMEOUT:
                    stats["timeout"] += 1
                    subfolder = "hangs"
                elif status == ExecutionStatus.CRASH:
                    stats["crash"] += 1
                    subfolder = "crash"
                    crashes.append(output)
                    logger.logo(f"--- CRASH FOUND ---: {msg}")
                    dump_code = f'"""\n{msg}\n"""\n{output}'
                elif status == ExecutionStatus.EXCEPTION:
                    stats["exception"] += 1
                    subfolder = "exception"
                    dump_code = f'"""\n{msg}\n"""\n{output}'

                if subfolder:
                    filename = os.path.join(args.folder, subfolder, f"{api}_{stats['generated']}.py")
                    with open(filename, "w", errors="ignore") as f:
                        f.write(dump_code)
                else: # Handle cases where subfolder is not set
                    filename = None


                if status == ExecutionStatus.SUCCESS:
                    round_valid += 1
                    generations.append(output)
                    filenames.append(filename)
                    if args.seed_selection_algo == "coverage":
                        _, new_coverage = mp_executor.coverate_run_status_mp(
                            output, args.library, cov_executor=mp_executor.cov_executor
                        )
                        add_flags.append(new_coverage)

            # --- FIX IS HERE ---
            # Call update with the correct number of arguments based on the algorithm.
            if args.seed_selection_algo == "coverage":
                ga.update(seed, generations, replace_type, r, filenames, add_flags)
            else:
                ga.update(seed, generations, replace_type, r, filenames)
            # --- END FIX ---

        num_valid += round_valid
        if round_valid == 0:
            mp_executor.test_executor.restart()

        generation_time.append(g_time)
        validation_time.append(v_time)
        total_programs.append(stats["generated"])
        r += 1
        logger.logo(f"--- New Valid: {round_valid} | Gen Time: {g_time:.2f}s | Val Time: {v_time:.2f}s ---")

        if model.backend == "hf":
            import torch
            torch.cuda.empty_cache()

        total_run_time.append(time.time() - start_time_total)

    logger.logo("-" * 20)
    logger.logo(f"Total valid outputs: {num_valid} using {sum(generation_time):.2f}s generation, {sum(validation_time):.2f}s validation")
    logger.logo(f"Stats: {stats}")
    logger.logo("-" * 20)

    return (
        ga.info_code, ga.get_p(), crashes, generation_time, validation_time,
        total_run_time, total_programs,
    )


def generate(args, model: LanguageModel):
    os.makedirs(args.folder, exist_ok=True)
    for sub in ["seed", "valid", "flaky", "hangs", "crash", "exception", "notarget"]:
        os.makedirs(os.path.join(args.folder, sub), exist_ok=True)

    with open(os.path.join(args.folder, "args.txt"), "w") as f:
        f.write(str(args))

    logger = Logger(os.path.join(os.path.dirname(__file__), args.folder))
    gen_ret = {}
    apis = get_initial_programs(
        args.seedfolder, model.infill_ph, args.library, "argument", target_api=args.api
    )

    if (args.api not in apis) and args.api != "all":
        logger.logo(f"Did not find {args.api} in list of valid seed apis")
        return

    for api, v in apis.items():
        if args.api != api and args.api != "all":
            continue
        if len(v) == 0:
            continue
        logger.logo(f"--- Generating for {api} --- | {len(v)} initial seeds")
        seeds_for_generation = []
        for idx, seed in enumerate(v):
            if not args.only_valid: # Take all seeds if not only_valid
                seeds_for_generation.append(seed["original"])
            else: # If only_valid, check status
                status, _ = validate_status(seed["original"], args.library, validate_mode=args.validate_mode, test_executor=mp_executor.test_executor)
                if status == ExecutionStatus.SUCCESS:
                    seeds_for_generation.append(seed["original"])

            with open(os.path.join(args.folder, "seed", f"{api}_seed{idx+1}.py"), "w") as f:
                f.write(seed["original"])

        if len(seeds_for_generation) > 0:
            gen_ret[api] = {"seeds": seeds_for_generation}
            (
                gen_ret[api]["outputs"], gen_ret[api]["p"], gen_ret[api]["crashes"],
                gen_ret[api]["g_time"], gen_ret[api]["v_time"], gen_ret[api]["tot_time"],
                gen_ret[api]["tot_prog"],
            ) = generate_loop(args, model, seeds_for_generation, api, logger, args.max_valid)

        mp_executor.test_executor.restart()
        if model.backend == "hf":
            import torch
            torch.cuda.empty_cache()

        with open(os.path.join(args.folder, "outputs.json"), "a") as f:
            f.write("\n")
            f.write(json.dumps({api: gen_ret.get(api, {})})) # Use .get for safety

    print("Generation process finished.")


def main():
    print("Current directory: ", os.getcwd())
    parser = argparse.ArgumentParser()
    # --- ARGUMENTS ARE KEPT IDENTICAL TO THE ORIGINAL FOR COMPATIBILITY ---
    parser.add_argument("--model_name", type=str, default="ollama/codegemma:7b", help="Model identifier. Use 'ollama/<model_name>' for local Ollama models or a Hugging Face path for legacy models.")
    parser.add_argument("--library", type=str, default=None, help="either 'torch' or 'tf'")
    parser.add_argument("--api", type=str, default=None)
    parser.add_argument("--apilist", type=str, default=None)
    parser.add_argument("--startid", type=int, default=0)
    parser.add_argument("--endid", type=int, default=-1)
    parser.add_argument("--folder", type=str, default="Result/test")
    parser.add_argument("--seedfolder", type=str, default="../codex_seed_programs/pt-codex/raw")
    parser.add_argument("--use_sample_apis", action="store_true", default=False)
    parser.add_argument("--random_seed", type=int, default=420)
    parser.add_argument("--max_valid", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=10, help="Number of samples to generate per seed. Mapped to num_samples for the model.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--seed_pool_size", type=int, default=30)
    parser.add_argument("--only_valid", action="store_true", default=False)
    parser.add_argument("--relaxargmut", action="store_true", default=False)
    parser.add_argument("--seed_selection_algo", type=str, default="random", choices=["fitness", "random", "coverage"])
    parser.add_argument("--mutator_selection_algo", type=str, default="epsgreedy", choices=["heuristic", "epsgreedy", "ucb", "random", "ts"])
    parser.add_argument("--use_single_mutator", action="store_true", default=False)
    parser.add_argument("--replace_type", type=str, default=None)
    parser.add_argument("--mutator_set", type=str, default="all", choices=["all", "noprefix", "nosuffix", "noargument", "nomethod"])
    parser.add_argument("--validate_mode", type=str, default="multiprocess", choices=["process", "multiprocess"])
    parser.add_argument("--close_fd_mask", type=int, default=1)
    args = parser.parse_args()

    if not args.library:
        raise ValueError("--library ('torch' or 'tf') is a required argument.")

    if args.api == "all":
        # This logic remains the same to support the original workflow
        run_args = ["python"] + argparse._sys.argv
        if args.apilist is not None:
            with open(args.apilist, "r") as f:
                all_apis = f.read().splitlines()
            if args.endid != -1:
                all_apis = all_apis[: args.endid]
            all_apis = all_apis[args.startid :]
        else:
            all_apis = load_apis(args.library, args.use_sample_apis)
        ind = run_args.index("all")
        for api_idx, api in enumerate(all_apis):
            print(f"[{api_idx + 1}/{len(all_apis)}] {api}")
            peek_seeds = glob.glob(os.path.join(args.seedfolder, f"{api}*.py")) # More robust glob
            if not peek_seeds:
                 peek_seeds = glob.glob(os.path.join(args.seedfolder, api, "*.py"))
            if not peek_seeds:
                print(f"---Skip {api} for lack of valid seed---")
                continue
            if os.path.exists(os.path.join(args.folder, "seed", f"{api}_seed1.py")):
                print(f"---Skip {api} because seed1.py already exists---")
                continue
            run_args_api = run_args.copy()
            run_args_api[ind] = api
            run_cmd(run_args_api, timeout=args.timeout + 300, verbose=True)
        exit(0)

    mp_executor.init_test_executor(args, cov=(args.seed_selection_algo == "coverage"))
    set_seed(args.random_seed)

    try:
        model = LanguageModel(args.model_name)
        generate(args, model)
    except Exception as e:
        import traceback
        print(f"An unhandled error occurred in main execution: {e}")
        traceback.print_exc()
    finally:
        mp_executor.kill_executors()


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()