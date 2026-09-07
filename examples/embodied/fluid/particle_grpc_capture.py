"""
本地 gRPC 服务：接收 OrcaSPH 发来的 ParticleFrame / RawParticleFrame，记录 total 粒子数与时间戳。

用于无 OrcaStudio 或端口冲突时，在本机统计 SPH 推送的粒子数量时间线（与 sph_sim_config 中
particle_render.grpc.server_address 一致）。
"""
from __future__ import annotations

import os
import threading
import time
from concurrent import futures
from typing import List, Optional, Tuple, Any

import grpc

# 由 orca-sph 包提供（与 SPH 侧用的同一套 proto）
from orcasph_client.protos import particle_data_pb2
from orcasph_client.protos import particle_data_pb2_grpc


class _ParticleCaptureServicer(particle_data_pb2_grpc.ParticleDataServiceServicer):
    def __init__(self, sink: "ParticleCountSink") -> None:
        self._sink = sink
        self._frame_debug = os.environ.get("ORCA_PLAYGROUND_DEBUG_GRPC_FRAME", "")

    def UpdateParticleFrame(self, request, context):
        t = time.perf_counter()
        n = 0
        if request.HasField("header"):
            n = int(request.header.total_particle_count)
        else:
            n = len(request.particles)
        self._sink.add_sample(t, n, "UpdateParticleFrame")
        if self._frame_debug and int(self._frame_debug or 0) > 0:
            if not hasattr(self, "_upf_count"):
                self._upf_count = 0
            self._upf_count += 1
            if self._upf_count <= 5:
                print(
                    f"[ParticleCapture] UpdateParticleFrame #{self._upf_count} n={n}",
                    flush=True,
                )
        return particle_data_pb2.ParticleFrameResponse(
            success=True,
            message="ok",
            particles_received=n,
        )

    def UpdateRawParticleFrame(self, request, context):
        t = time.perf_counter()
        n = 0
        for b in request.batches:
            n += int(b.particle_count)
        self._sink.add_sample(t, n, "UpdateRawParticleFrame")
        if self._frame_debug and int(self._frame_debug or 0) > 0:
            print(f"[ParticleCapture] UpdateRawParticleFrame n={n}", flush=True)
        return particle_data_pb2.RawParticleFrameResponse(
            success=True,
            message="ok",
            particles_received=n,
        )

    def EndSimulation(self, request, context):
        return particle_data_pb2.EndSimulationResponse(success=True, message="ok")

    def GetRenderedParticleFrame(self, request, context):
        return particle_data_pb2.GetRenderedParticleFrameResponse(
            rendered_particle_frame_id=0
        )

    def UpdateParticles(self, request, context):
        t = time.perf_counter()
        n = len(request.particles) if request.particles else 0
        self._sink.add_sample(t, n, "UpdateParticles")
        return particle_data_pb2.ParticleDataResponse(success=True, message="ok")

    def UpdateVoxelGrid(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        return particle_data_pb2.VoxelGridDataResponse()


class ParticleCountSink:
    """线程安全：累积 (perf_counter, count, method) 样本。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.samples: List[Tuple[float, int, str]] = []

    def add_sample(self, t: float, count: int, method: str) -> None:
        with self._lock:
            self.samples.append((t, int(count), method))

    def snapshot(self) -> List[Tuple[float, int, str]]:
        with self._lock:
            return list(self.samples)


def _server_options() -> List[Tuple[str, Any]]:
    # 与 SPH 侧 I-Frame 等大包对齐，避免在反序列化前被 gRPC 拒绝（导致 Python 收不到任何帧）
    mb = 64
    m = mb * 1024 * 1024
    return [
        ("grpc.max_send_message_length", m),
        ("grpc.max_receive_message_length", m),
    ]


def start_particle_grpc_capture_server(
    port: int, max_workers: int = 32
) -> Tuple[grpc.Server, ParticleCountSink]:
    """
    在 0.0.0.0:port 上启动捕获服务。调用方在跑完仿真后应 server.stop(0)。
    """
    sink = ParticleCountSink()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers), options=_server_options()
    )
    particle_data_pb2_grpc.add_ParticleDataServiceServicer_to_server(
        _ParticleCaptureServicer(sink), server
    )
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    return server, sink


def resample_to_intervals(
    samples: List[Tuple[float, int, str]],
    interval_s: float,
    duration_s: float,
) -> Tuple[List[float], List[int]]:
    """
    以首帧时间为 t0，在 t0, t0+dt, t0+2dt, ... 直到 t0+duration_s 上，
    取「时间不超过该点」的最后一个 count（阶跃保持）。
    若某时尚无样本，用 0（或此前最近值在循环中处理）。
    """
    if not samples:
        return (
            [round(i * interval_s, 6) for i in range(int(duration_s / interval_s) + 1)],
            [0] * (int(duration_s / interval_s) + 1),
        )
    t0 = samples[0][0]
    # 仅按 (t, n) 重采样，按时间排序
    points: List[Tuple[float, int]] = sorted(
        ((s[0], s[1]) for s in samples), key=lambda p: p[0]
    )
    times_rel: List[float] = []
    counts: List[int] = []
    n_steps = int(round(duration_s / interval_s))
    for k in range(n_steps + 1):
        target = t0 + k * interval_s
        last = 0
        for t, c in points:
            if t <= target:
                last = c
            else:
                break
        times_rel.append(round(k * interval_s, 6))
        counts.append(last)
    return times_rel, counts
