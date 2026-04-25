"""
SSH Client Wrapper

Provides SSH connectivity to the remote GPU node for executing commands,
reading files, and writing files on the vLLM host.

SOURCE: ai-perf-hackathon/agent/ssh_client.py (verbatim)

Key class:
    - SSHClient(host, user, key_path)
        - run(command, timeout) → stdout, stderr, exit_code
        - read_file(path) → contents
        - write_file(path, content) → success
        - upload_file(local_path, remote_path) → success
        - close()

Target host: NVIDIA cloud GPU (aansharm-0-yxg5)
"""

# TODO: Copy SSHClient class verbatim from source
