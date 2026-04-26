"""
Pod Manager for Isolated Pod-per-Experiment Architecture.

Creates ephemeral vLLM pods from a YAML template for each tuning experiment.
The baseline pod is never modified; experiment pods are created with different
vLLM launch args, benchmarked, compared against baseline, then deleted.

Usage:
    pm = PodManager(namespace="toa-hack", kubeconfig=None, base_pod_yaml_path="aanya-pod.yaml")
    pod_name, endpoint = pm.create_pod(["--enable-chunked-prefill", "--gpu-memory-utilization", "0.95"])
    # ... run benchmarks against endpoint ...
    pm.delete_pod(pod_name)
"""

from __future__ import annotations

import copy
import os
import signal
import subprocess
import tempfile
import time
from typing import Optional

import yaml


class PodManager:
    """Manages ephemeral vLLM experiment pods on OpenShift.

    Creates pods from a YAML template with modified vLLM args, waits for
    readiness, sets up port-forwarding, and cleans up on deletion.

    Parameters
    ----------
    namespace : str
        Kubernetes/OpenShift namespace.
    kubeconfig : str or None
        Path to kubeconfig. If None, uses default (~/.kube/config or $KUBECONFIG).
    base_pod_yaml_path : str
        Path to the pod YAML template (e.g. aanya-pod.yaml).
    base_port : int
        First local port to assign for port-forwarding experiment pods.
    """

    def __init__(
        self,
        namespace: str,
        kubeconfig: Optional[str] = None,
        base_pod_yaml_path: str = "aanya-pod.yaml",
        base_port: int = 8001,
    ):
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.base_yaml_path = base_pod_yaml_path
        self.base_port = base_port
        self.active_pods: dict[str, dict] = {}  # pod_name -> {"port_forward_proc": ..., "local_port": ...}
        self.active_benchmark_pods: dict[str, dict] = {}  # pod_name -> metadata
        self._next_port = base_port
        self.disable_prefix_caching = True  # default: disable on experiment pods

        # Load and validate the template once
        with open(self.base_yaml_path, "r") as f:
            self._template = yaml.safe_load(f)

        if self._template.get("kind") != "Pod":
            raise ValueError(f"Template {base_pod_yaml_path} is not a Pod manifest (kind={self._template.get('kind')})")

    def _build_oc_base(self) -> list[str]:
        """Build the base ``oc`` command with kubeconfig and namespace."""
        cmd = ["oc"]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        cmd += ["-n", self.namespace]
        return cmd

    def _generate_pod_name(self) -> str:
        """Generate a unique pod name based on timestamp."""
        ts = int(time.time())
        return f"vllm-tune-{ts}"

    def _build_pod_manifest(self, pod_name: str, vllm_args: list[str]) -> dict:
        """Create a pod manifest from the template with extra vLLM args.

        Appends ``vllm_args`` to the existing ``args`` list of the first
        container (assumed to be the vLLM container).
        """
        manifest = copy.deepcopy(self._template)

        # Set unique pod name
        manifest["metadata"]["name"] = pod_name

        # Add a label to identify experiment pods for easy cleanup
        labels = manifest["metadata"].setdefault("labels", {})
        labels["vllm-experiment"] = "true"

        # Append tuning args to the container's args list
        container = manifest["spec"]["containers"][0]
        existing_args = list(container.get("args", []))
        # Disable prefix caching by default for fair comparison (cold cache)
        if self.disable_prefix_caching and "--no-enable-prefix-caching" not in vllm_args:
            existing_args.append("--no-enable-prefix-caching")
        existing_args.extend(vllm_args)
        container["args"] = existing_args

        return manifest

    def _wait_for_ready(self, pod_name: str, timeout: int = 120, poll_interval: int = 5) -> bool:
        """Poll pod readiness until ready or timeout.

        Returns True if the pod reached Running + Ready state.
        """
        oc_base = self._build_oc_base()
        deadline = time.time() + timeout

        while time.time() < deadline:
            # Check pod phase
            cmd = oc_base + [
                "get", "pod", pod_name,
                "-o", "jsonpath={.status.phase}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            phase = result.stdout.strip()

            if phase == "Failed" or phase == "Unknown":
                # Pod failed to start — get events for debugging
                events_cmd = oc_base + [
                    "get", "events",
                    "--field-selector", f"involvedObject.name={pod_name}",
                    "--sort-by=.lastTimestamp",
                ]
                events_result = subprocess.run(events_cmd, capture_output=True, text=True, timeout=15)
                raise RuntimeError(
                    f"Pod {pod_name} entered phase '{phase}'. Events:\n{events_result.stdout}"
                )

            if phase == "Running":
                # Check readiness condition
                ready_cmd = oc_base + [
                    "get", "pod", pod_name,
                    "-o", "jsonpath={.status.conditions[?(@.type=='Ready')].status}",
                ]
                ready_result = subprocess.run(ready_cmd, capture_output=True, text=True, timeout=15)
                if ready_result.stdout.strip() == "True":
                    return True

            time.sleep(poll_interval)

        raise TimeoutError(f"Pod {pod_name} not ready after {timeout}s (last phase: {phase})")

    def _start_port_forward(self, pod_name: str, local_port: int, remote_port: int = 8000) -> subprocess.Popen:
        """Start ``oc port-forward`` as a background subprocess."""
        cmd = self._build_oc_base() + [
            "port-forward", pod_name, f"{local_port}:{remote_port}",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Give port-forward a moment to bind
        time.sleep(2)

        # Check it didn't immediately die
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(
                f"oc port-forward for {pod_name} exited immediately (rc={proc.returncode}): {stderr}"
            )

        return proc

    def create_pod(self, vllm_args: list[str]) -> tuple[str, str]:
        """Create an experiment pod with extra vLLM args, wait for readiness,
        and start port-forwarding.

        Parameters
        ----------
        vllm_args : list[str]
            Extra CLI args for vLLM (e.g. ["--enable-chunked-prefill",
            "--gpu-memory-utilization", "0.95"]).

        Returns
        -------
        tuple[str, str]
            (pod_name, endpoint_url) e.g. ("vllm-tune-1714000000", "http://localhost:8001")
        """
        pod_name = self._generate_pod_name()
        local_port = self._next_port
        self._next_port += 1

        print(f">> PodManager: Creating pod {pod_name} with args {vllm_args}", flush=True)

        # Build manifest
        manifest = self._build_pod_manifest(pod_name, vllm_args)

        # Write to temp file and apply
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"{pod_name}_", delete=False
        ) as tmp:
            yaml.dump(manifest, tmp, default_flow_style=False)
            tmp_path = tmp.name

        try:
            apply_cmd = self._build_oc_base() + ["apply", "-f", tmp_path]
            result = subprocess.run(apply_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"oc apply failed: {result.stderr}")
            print(f"   Pod {pod_name} created.", flush=True)
        finally:
            os.unlink(tmp_path)

        # Wait for pod to be ready
        print(f"   Waiting for pod {pod_name} to be ready...", flush=True)
        self._wait_for_ready(pod_name)
        print(f"   Pod {pod_name} is ready.", flush=True)

        # Start port-forward
        print(f"   Starting port-forward :{local_port} -> {pod_name}:8000", flush=True)
        pf_proc = self._start_port_forward(pod_name, local_port)
        print(f"   Port-forward active (pid={pf_proc.pid}).", flush=True)

        endpoint = f"http://localhost:{local_port}"

        self.active_pods[pod_name] = {
            "port_forward_proc": pf_proc,
            "local_port": local_port,
            "vllm_args": vllm_args,
            "endpoint": endpoint,
        }

        return pod_name, endpoint

    def create_profiled_pod(
        self,
        vllm_args: list[str],
        profiler_config,
    ) -> tuple[str, str]:
        """Create experiment pod with profiler annotations for webhook injection.

        Same as create_pod but overlays profiler annotations so the
        mutating webhook injects PyTorch profiler instrumentation.
        """
        pod_name = self._generate_pod_name()
        local_port = self._next_port
        self._next_port += 1

        print(f">> PodManager: Creating profiled pod {pod_name}", flush=True)
        print(f"   vLLM args: {vllm_args}", flush=True)
        print(f"   Profiler ranges: {profiler_config.ranges}", flush=True)

        manifest = self._build_pod_manifest(pod_name, vllm_args)

        # Add label for webhook matching (only profiled pods get instrumented)
        labels = manifest["metadata"].setdefault("labels", {})
        labels["managed-by"] = "profiler-agent"

        annotations = manifest["metadata"].setdefault("annotations", {})
        annotations.update(profiler_config.to_annotations())

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"{pod_name}_", delete=False
        ) as tmp:
            yaml.dump(manifest, tmp, default_flow_style=False)
            tmp_path = tmp.name

        try:
            apply_cmd = self._build_oc_base() + ["apply", "-f", tmp_path]
            result = subprocess.run(apply_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"oc apply failed: {result.stderr}")
            print(f"   Pod {pod_name} created with profiler annotations.", flush=True)
        finally:
            os.unlink(tmp_path)

        print(f"   Waiting for pod {pod_name} to be ready...", flush=True)
        self._wait_for_ready(pod_name, timeout=300)
        print(f"   Pod {pod_name} is ready.", flush=True)

        print(f"   Starting port-forward :{local_port} -> {pod_name}:8000", flush=True)
        pf_proc = self._start_port_forward(pod_name, local_port)
        print(f"   Port-forward active (pid={pf_proc.pid}).", flush=True)

        endpoint = f"http://localhost:{local_port}"

        self.active_pods[pod_name] = {
            "port_forward_proc": pf_proc,
            "local_port": local_port,
            "vllm_args": vllm_args,
            "endpoint": endpoint,
            "profiler_config": profiler_config,
        }

        return pod_name, endpoint

    def delete_pod(self, pod_name: str) -> None:
        """Delete an experiment pod and kill its port-forward process.

        Parameters
        ----------
        pod_name : str
            Name of the pod to delete.
        """
        info = self.active_pods.pop(pod_name, None)

        # Kill port-forward
        if info and info.get("port_forward_proc"):
            proc = info["port_forward_proc"]
            try:
                os.kill(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
            print(f"   Port-forward for {pod_name} killed.", flush=True)

        # Delete pod
        delete_cmd = self._build_oc_base() + ["delete", "pod", pod_name, "--grace-period=0", "--force"]
        result = subprocess.run(delete_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"   Pod {pod_name} deleted.", flush=True)
        else:
            print(f"   Warning: Failed to delete pod {pod_name}: {result.stderr}", flush=True)

    def cleanup_all(self) -> None:
        """Delete all active experiment and benchmark pods. Called at agent exit."""
        pod_names = list(self.active_pods.keys())
        bench_names = list(self.active_benchmark_pods.keys())

        if not pod_names and not bench_names:
            return

        total = len(pod_names) + len(bench_names)
        print(f">> PodManager: Cleaning up {total} pod(s)...", flush=True)
        for pod_name in pod_names:
            try:
                self.delete_pod(pod_name)
            except Exception as e:
                print(f"   Warning: cleanup failed for {pod_name}: {e}", flush=True)
        for pod_name in bench_names:
            try:
                self.delete_benchmark_pod(pod_name)
            except Exception as e:
                print(f"   Warning: cleanup failed for {pod_name}: {e}", flush=True)

    # ── Benchmark Pod Management ───────────────────────────────────────────

    GUIDELLM_IMAGE = "ghcr.io/vllm-project/guidellm:v0.5.1"

    def _get_pod_ip(self, pod_name: str) -> str:
        """Get the cluster IP of a running pod."""
        cmd = self._build_oc_base() + [
            "get", "pod", pod_name,
            "-o", "jsonpath={.status.podIP}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"Cannot get IP for pod {pod_name}: {result.stderr}")
        return result.stdout.strip()

    def create_benchmark_pod(
        self,
        target_pod_name: str,
        model: str,
        prompt_tokens: int,
        output_tokens: int,
        concurrency: str,
        max_seconds: int = 30,
    ) -> str:
        """Create a GuideLLM benchmark pod targeting a vLLM pod.

        Returns the benchmark pod name.
        """
        pod_ip = self._get_pod_ip(target_pod_name)
        target_url = f"http://{pod_ip}:8000"

        # GuideLLM uses --model for both API requests and HF tokenizer lookup
        hf_model = model
        if hf_model.startswith("/models/"):
            hf_model = hf_model[len("/models/"):]

        ts = int(time.time())
        bench_pod_name = f"guidellm-bench-{ts}"

        manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": bench_pod_name,
                "namespace": self.namespace,
                "labels": {
                    "app": "guidellm-bench",
                    "vllm-experiment": "true",
                },
            },
            "spec": {
                "restartPolicy": "Never",
                "containers": [{
                    "name": "guidellm",
                    "image": self.GUIDELLM_IMAGE,
                    "command": ["/bin/sh", "-c",
                        " ".join([
                            "guidellm", "benchmark",
                            "--target", target_url,
                            "--model", hf_model,
                            "--profile", "concurrent",
                            "--rate", concurrency,
                            f"--data", f"prompt_tokens={prompt_tokens},output_tokens={output_tokens}",
                            "--max-seconds", str(max_seconds),
                            "--request-type", "text_completions",
                            "--output-dir", "/tmp/results",
                            "--outputs", "benchmark.json",
                        ])
                        + " && echo 'BENCHMARK_DONE' && sleep 300"
                    ],
                    "env": [{"name": "HF_HOME", "value": "/tmp/hf-cache"}],
                    "volumeMounts": [
                        {"name": "hf-cache", "mountPath": "/tmp/hf-cache"},
                    ],
                    "resources": {
                        "requests": {"cpu": "1", "memory": "2Gi"},
                        "limits": {"cpu": "2", "memory": "4Gi"},
                    },
                }],
                "volumes": [
                    {"name": "hf-cache", "emptyDir": {}},
                ],
            },
        }

        print(f">> PodManager: Creating benchmark pod {bench_pod_name}", flush=True)
        print(f"   Target: {target_url} (pod {target_pod_name})", flush=True)
        print(f"   Concurrency: {concurrency}, ISL={prompt_tokens}, OSL={output_tokens}", flush=True)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"{bench_pod_name}_", delete=False
        ) as tmp:
            yaml.dump(manifest, tmp, default_flow_style=False)
            tmp_path = tmp.name

        try:
            apply_cmd = self._build_oc_base() + ["apply", "-f", tmp_path]
            result = subprocess.run(apply_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"oc apply failed: {result.stderr}")
        finally:
            os.unlink(tmp_path)

        self.active_benchmark_pods[bench_pod_name] = {
            "target_pod": target_pod_name,
            "target_url": target_url,
        }

        return bench_pod_name

    def wait_for_pod_completion(self, pod_name: str, timeout: int = 600, poll_interval: int = 10) -> bool:
        """Wait for benchmark pod to finish.

        The pod runs `guidellm ... && echo BENCHMARK_DONE && sleep 300`,
        so it stays Running after completion. We poll logs for the marker.
        """
        oc_base = self._build_oc_base()
        deadline = time.time() + timeout

        while time.time() < deadline:
            phase_cmd = oc_base + [
                "get", "pod", pod_name,
                "-o", "jsonpath={.status.phase}",
            ]
            phase_result = subprocess.run(phase_cmd, capture_output=True, text=True, timeout=15)
            phase = phase_result.stdout.strip()

            if phase == "Failed":
                logs = self._get_pod_logs(pod_name, tail=50)
                raise RuntimeError(f"Benchmark pod {pod_name} failed.\nLogs:\n{logs}")

            if phase == "Succeeded":
                print(f"   Benchmark pod {pod_name} completed.", flush=True)
                return True

            if phase == "Running":
                logs = self._get_pod_logs(pod_name, tail=5)
                if "BENCHMARK_DONE" in logs:
                    print(f"   Benchmark pod {pod_name} completed.", flush=True)
                    return True

            time.sleep(poll_interval)

        raise TimeoutError(f"Benchmark pod {pod_name} not done after {timeout}s (phase: {phase})")

    def copy_results_from_pod(self, bench_pod_name: str, remote_path: str, local_path: str) -> str:
        """Copy results from the benchmark pod (still Running due to post-benchmark sleep)."""
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

        cmd = self._build_oc_base() + [
            "cp", f"{bench_pod_name}:{remote_path}", local_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"oc cp failed: {result.stderr}")

        print(f"   Results copied to {local_path}", flush=True)
        return local_path

    def delete_benchmark_pod(self, pod_name: str) -> None:
        """Delete a benchmark pod."""
        self.active_benchmark_pods.pop(pod_name, None)

        delete_cmd = self._build_oc_base() + ["delete", "pod", pod_name, "--grace-period=0"]
        result = subprocess.run(delete_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"   Benchmark pod {pod_name} deleted.", flush=True)
        else:
            print(f"   Warning: Failed to delete benchmark pod {pod_name}: {result.stderr}", flush=True)

    def _get_pod_logs(self, pod_name: str, tail: int = 50) -> str:
        """Get last N lines of pod logs."""
        cmd = self._build_oc_base() + ["logs", pod_name, f"--tail={tail}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return result.stdout if result.returncode == 0 else result.stderr

    def get_active_pods(self) -> dict[str, dict]:
        """Return info about active experiment pods."""
        return dict(self.active_pods)
