#!/usr/bin/env bash
# 批跑全部 beginner 课程（自动发现，无需维护清单）：
# 逐课启动，每课驻留 N 秒供视口观察，自动进入下一课。
#
# 用法:
#   bash tools/run_all_beginner_lessons.sh [每课秒数，默认 20]
#   ORCA_PYTHON=/path/to/python bash tools/run_all_beginner_lessons.sh
#
# 前置: OrcaLab 已运行（默认 localhost:50051）、资产包已导入。
# 原理: 扫描 examples/euler/beginner/stage*/lesson_*/run.py 自动发现课程，
#       按课号排序；每课驻留到时后发 SIGINT，课程自带的 KeyboardInterrupt
#       分支优雅退出（scene.close()），不会残留场景连接。
# 特例: 第 13 课起自动带 --default-scene（批跑模式下不等待手动拖拽，
#       层 3 课均自行退出，超时仅作兜底）。
# 中断: 观察中途想停全流程，连按 Ctrl+C 两次。

set -u

DUR="${1:-20}"
ADDR="${ORCAGYM_ADDR:-localhost:50051}"
PY="${ORCA_PYTHON:-python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 自动发现全部课程模块并按课号排序
LESSONS=($(cd "$ROOT" && ls examples/euler/beginner/stage*/lesson_*/run.py 2>/dev/null \
  | sed -E 's|examples/euler/beginner/||; s|/run\.py$||; s|/|.|g' \
  | sort -t_ -k2 -n))
if [ "${#LESSONS[@]}" -eq 0 ]; then
  echo "未发现任何课程（examples/euler/beginner/stage*/lesson_*/run.py）" >&2
  exit 1
fi
echo "发现 ${#LESSONS[@]} 课："
printf '  %s\n' "${LESSONS[@]}"

FAIL=0
for lesson in "${LESSONS[@]}"; do
  num="$(echo "$lesson" | grep -o 'lesson_[0-9]*' | grep -o '[0-9]*')"
  extra_args=()
  if [ "$num" -ge 13 ]; then
    extra_args+=(--default-scene)
  fi

  echo ""
  echo "=================================================================="
  echo ">>> 第 ${num} 课（驻留 ${DUR}s 后自动进入下一课，Ctrl+C 中断全流程）"
  echo "=================================================================="
  if ! timeout --signal=INT --kill-after=10 "${DUR}s" \
      "$PY" -m "$lesson" --addr "$ADDR" "${extra_args[@]+"${extra_args[@]}"}"; then
    rc=$?
    # 124/137 = 超时正常翻页；其余码记录失败但继续
    if [ "$rc" -ne 124 ] && [ "$rc" -ne 137 ]; then
      echo ">>> 第 ${num} 课异常退出（rc=${rc}），已记录，继续下一课"
      FAIL=$((FAIL + 1))
    fi
  fi
done

echo ""
echo "=================================================================="
if [ "$FAIL" -eq 0 ]; then
  echo "全部 ${#LESSONS[@]} 课跑完，无异常退出"
else
  echo "跑完，但有 ${FAIL} 课异常退出，请回看上方日志"
fi
echo "=================================================================="
