# openpi 自定义修改说明

本目录存放 `examples/so101` 示例依赖的 openpi patch。

使用方式：

```bash
git clone https://github.com/openverse-orca/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
git apply /path/to/OrcaPlayground/examples/so101/openpi_patches/so101_openpi.patch
```

应用后即可按 `examples/so101/README.md` 中的方式启动：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_h7_lora \
    --policy.dir=../OrcaGym-SO101/models/h11_lora/6000
```
