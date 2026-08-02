# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for lustre->node-local model staging (model.stage_dir)."""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from srtctl.backends import TRTLLMProtocol, TRTLLMServerConfig
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig


def _runtime(*, staged=None, hf=False, model="/lustre/DeepSeek-V4-Pro"):
    return RuntimeContext(
        job_id="1",
        run_name="r",
        nodes=Nodes(head="n0", bench="n0", infra="n0", worker=("n1", "n2")),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=Path("/tmp/logs"),
        model_path=Path(model),
        container_image=Path("/img.sqsh"),
        gpus_per_node=4,
        network_interface="eth0",
        is_hf_model=hf,
        staged_model_path=(Path(staged) if staged else None),
    )


class TestWorkerModelArg:
    def test_default_is_model_mount(self):
        assert _runtime().worker_model_arg == "/model"

    def test_staged_path_wins(self):
        rt = _runtime(staged="/raid/scratch/models/DeepSeek-V4-Pro")
        assert rt.worker_model_arg == "/raid/scratch/models/DeepSeek-V4-Pro"

    def test_hf_uses_model_id(self):
        rt = _runtime(hf=True, model="deepseek-ai/DeepSeek-V4-Pro")
        assert rt.worker_model_arg == "deepseek-ai/DeepSeek-V4-Pro"


class TestSchema:
    def test_stage_dir_loads(self):
        data = {
            "name": "stage-test",
            "model": {
                "path": "/lustre/DeepSeek-V4-Pro",
                "container": "trtllm",
                "precision": "fp4",
                "stage_dir": "/raid/scratch/models",
            },
            "resources": {"gpu_type": "gb300", "gpus_per_node": 4, "agg_nodes": 1, "agg_workers": 1},
            "backend": {"type": "trtllm"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = SrtConfig.from_yaml(Path(f.name))
        assert config.model.stage_dir == "/raid/scratch/models"

    def test_stage_dir_defaults_none(self):
        data = {
            "name": "no-stage",
            "model": {"path": "/lustre/m", "container": "trtllm", "precision": "fp4"},
            "resources": {"gpu_type": "gb300", "gpus_per_node": 4, "agg_nodes": 1, "agg_workers": 1},
            "backend": {"type": "trtllm"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = SrtConfig.from_yaml(Path(f.name))
        assert config.model.stage_dir is None


class TestStageModelScript:
    @staticmethod
    def _run(script: Path, source: Path, destination: Path) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["STAGE_PARALLEL"] = "2"
        return subprocess.run(
            ["bash", str(script), str(source), str(destination)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_cold_stage_then_manifest_hit(self, tmp_path):
        script = Path(__file__).parents[1] / "src/srtctl/runtime_scripts/stage_model.sh"
        source = tmp_path / "source"
        destination = tmp_path / "cache" / "model"
        source.mkdir()
        (source / "config.json").write_text('{"model": "test"}\n')
        (source / "nested").mkdir()
        (source / "nested" / "weights.bin").write_bytes(b"weights")

        cold = self._run(script, source, destination)
        assert cold.returncode == 0, cold.stderr
        assert "manifest verified" in cold.stdout
        assert (destination / "config.json").read_text() == '{"model": "test"}\n'
        assert (destination / "nested" / "weights.bin").read_bytes() == b"weights"
        assert (destination / ".srtctl-stage-complete").is_file()

        hit = self._run(script, source, destination)
        assert hit.returncode == 0, hit.stderr
        assert "manifest hit" in hit.stdout

    def test_same_size_source_change_replaces_generation(self, tmp_path):
        script = Path(__file__).parents[1] / "src/srtctl/runtime_scripts/stage_model.sh"
        source = tmp_path / "source"
        destination = tmp_path / "cache" / "model"
        source.mkdir()
        weight = source / "weights.bin"
        weight.write_bytes(b"before")

        first = self._run(script, source, destination)
        assert first.returncode == 0, first.stderr
        first_marker = (destination / ".srtctl-stage-complete").read_text()

        weight.write_bytes(b"after!")
        stat = weight.stat()
        os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        second = self._run(script, source, destination)

        assert second.returncode == 0, second.stderr
        assert "manifest verified" in second.stdout
        assert "manifest hit" not in second.stdout
        assert (destination / "weights.bin").read_bytes() == b"after!"
        assert (destination / ".srtctl-stage-complete").read_text() != first_marker

    def test_concurrent_stage_is_serialized(self, tmp_path):
        script = Path(__file__).parents[1] / "src/srtctl/runtime_scripts/stage_model.sh"
        source = tmp_path / "source"
        destination = tmp_path / "cache" / "model"
        source.mkdir()
        for index in range(32):
            (source / f"weight-{index:02d}.bin").write_bytes(bytes([index]) * 4096)

        env = os.environ.copy()
        env["STAGE_PARALLEL"] = "2"
        command = ["bash", str(script), str(source), str(destination)]
        first = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        second = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        first_stdout, first_stderr = first.communicate(timeout=30)
        second_stdout, second_stderr = second.communicate(timeout=30)

        assert first.returncode == 0, first_stderr
        assert second.returncode == 0, second_stderr
        combined = first_stdout + second_stdout
        assert combined.count("manifest verified") == 1
        assert combined.count("manifest hit") == 1
        assert len(list(destination.glob("weight-*.bin"))) == 32

    def test_overlapping_destination_is_rejected(self, tmp_path):
        script = Path(__file__).parents[1] / "src/srtctl/runtime_scripts/stage_model.sh"
        source = tmp_path / "source"
        source.mkdir()
        (source / "weights.bin").write_bytes(b"weights")

        result = self._run(script, source, source / "nested-cache")

        assert result.returncode == 2
        assert "must not overlap" in result.stderr
        assert (source / "weights.bin").read_bytes() == b"weights"


class TestWorkerCommandUsesStagedPath:
    def _proc(self):
        from srtctl.core.topology import Process

        return Process(
            node="n1",
            gpu_indices=frozenset([0]),
            sys_port=8081,
            http_port=6100,
            endpoint_mode="decode",
            endpoint_index=0,
            node_rank=0,
        )

    def _runtime_mock(self, tmp_path, staged_arg):
        rt = MagicMock()
        rt.worker_model_arg = staged_arg
        rt.is_hf_model = False
        rt.model_path = Path("/lustre/DeepSeek-V4-Pro")
        rt.log_dir = Path(tmp_path)
        return rt

    def test_trtllm_serve_worker_uses_staged_path(self, tmp_path):
        backend = TRTLLMProtocol(trtllm_config=TRTLLMServerConfig(decode={"tensor_parallel_size": 4}))
        cmd = backend.build_worker_command(
            self._proc(),
            [self._proc()],
            self._runtime_mock(tmp_path, "/raid/scratch/models/DeepSeek-V4-Pro"),
            frontend_type="trtllm_serve",
        )
        assert "/raid/scratch/models/DeepSeek-V4-Pro" in cmd
        assert "/model" not in cmd

    def test_dynamo_worker_uses_staged_path(self, tmp_path):
        backend = TRTLLMProtocol(trtllm_config=TRTLLMServerConfig(decode={"tensor_parallel_size": 4}))
        cmd = backend.build_worker_command(
            self._proc(),
            [self._proc()],
            self._runtime_mock(tmp_path, "/raid/scratch/models/DeepSeek-V4-Pro"),
            frontend_type="dynamo",
        )
        # dynamo path passes it as --model-path
        assert "/raid/scratch/models/DeepSeek-V4-Pro" in cmd

    def test_dynamo_worker_does_not_publish_events_by_default(self, tmp_path):
        backend = TRTLLMProtocol(trtllm_config=TRTLLMServerConfig(decode={"tensor_parallel_size": 4}))
        cmd = backend.build_worker_command(
            self._proc(),
            [self._proc()],
            self._runtime_mock(tmp_path, "/raid/scratch/models/DeepSeek-V4-Pro"),
            frontend_type="dynamo",
        )
        assert "--publish-events-and-metrics" not in cmd

    def test_dynamo_worker_publish_events_enabled(self, tmp_path):
        backend = TRTLLMProtocol(
            trtllm_config=TRTLLMServerConfig(decode={"tensor_parallel_size": 4}),
            publish_events_and_metrics=True,
        )
        cmd = backend.build_worker_command(
            self._proc(),
            [self._proc()],
            self._runtime_mock(tmp_path, "/raid/scratch/models/DeepSeek-V4-Pro"),
            frontend_type="dynamo",
        )
        assert "--publish-events-and-metrics" in cmd
