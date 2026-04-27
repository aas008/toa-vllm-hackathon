There are 4 GPUs available to us on node 02. GPU: 4,5,6,7

I started a vllm server using the following cmd:
```
CUDA_VISIBLE_DEVICES=7 /home/lab/rawhad/vllm_venv/bin/vllm serve Qwen/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --host 0.0.0.0 \
  --port 8100 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

And ran a guidellm benchmark, using the following cmd:
```
!mkdir -p benchmark_results/test-run/
uv run guidellm benchmark run \
  --target "http://localhost:8100" \
  --model "qwen3-0.6b" \
  --processor "Qwen/Qwen3-0.6B" \
  --rate-type concurrent \
  --rate 20 \
  --max-seconds 10 \
  --data "prompt_tokens=2048,output_tokens=256" \
  --output-dir benchmark_results/test-run/prefill_heavy.json \
  --outputs json 2>&1 | tee benchmark_results/test-run/benchmarks_console.log
```


I need you to run 50 experiments and work towards finding the best set of cli args for vllm server deployment,
that will provide the best results.
You can run `score` script to get the final `goodput` score.
This matters the most, and you can get it by running:
```
uv run score.py benchmark_results/test-run/prefill_heavy.json
```

`goodput` metric is what you will be tracking, and your job is to make that go up!

> We ran an earlier test version without the skills and the report for that is in `report-run-1.md`. I renamed it to add suffix `-run-1`

I have added docs on vllm args and optimization methods. I would like you to refer to those when you are analyzing the results and hypothesizing, and designing next experiment setup.
Make sure you refer to those docs before you run another experiment. The docs are at `docs/**/*.md`

---

### Experiment Approach.

- After every experiment, I would like you to do a RCA on the benchmark results and vllm deployment, and come up with a hypothesis on why the server is slow, and a possible solution on how to increase the goodput tok/sec score.
- Then you deploy the vllm server with the new solution that you found, which will be your new experiment to validate your hypothesis. Null hypothesis!
- And this way you will make progress towards better and better configurations.
- While developing hypothesis, you will ofcourse look at more than just `goodput` score. You have a whole suite of scores available in console log, and json output.

---

### Changing vllm deployment configuration.

- You can edit the server config section in `deploy.sh` file.
- After changing the config, commit the file, and then run it to deploy the vllm server on the node.
- After deployment run the `guidellm` benchmark and `score.py` to get results.
- Reason, and write down your analysis and next steps in the report.
- Kill the vllm deployment using `kill.sh` script.
- Repeat for 50 experiments, and then give your final conclusion in the report too.

---

### Things to keep in mind.

1. You are the orchestrator and reasoning engine behind all of this. You should use subagents for executing tasks such as deploying vllm server, running guidellm benchmark, running the scoring function, analyzing and giving a high level overview of what all things that the guidellm run outputted.
2. The `guidellm` script might freeze sometimes, to remedy that you will have to either restart the benchmark or restart the server and benchmark.
3. You will have to report all your experiments and their results in a `.md` file in cwd.
4. The benchmark_results/ dir will have differente sub dirs for each experiment. eg: benchmark_results/exp_001/prefill_heavy.json.
5. The experiments that you will undertake, will have to be sequential.
6. Port will always be 8100. And CUDA_VISIBLE_DEVICES=4,5,6,7. Thats it. You dont have to reserve gpus or anything. Node will always be 02.
7. You can only edit `deploy.sh` and `report.md`. `report.md` will contain the final report.
8. You will be given a cc/** git branch. Use that to maintain sanity across file edits.

---

### Additional Information

- You can use [vllm-recipes](https://docs.vllm.ai/projects/recipes/en/latest/index.html) for reference on how all we can optimize the deployment of our model.
- Dont just blindly follow the recipes though. I want you to understand why a certain choice or argument was mentioned, and argue for why it fits our usecase, and how it solves our current bottleneck on the deployment side.

---

Start by getting a baseline from the last experiment run, and then run at least 50 experiments.
I am going to sleep right now, and I want you to run these experiments overnight, and so i can have a final report by the time I wake up in the morning.
Thank you :D
