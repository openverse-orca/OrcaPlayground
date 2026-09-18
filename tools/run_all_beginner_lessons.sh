#!/usr/bin/env bash
# 批跑 beginner 01–13 课：逐课启动，每课驻留 N 秒供视口观察，自动进入下一课。
#
# 用法:
#   bash tools/run_all_beginner_lessons.sh [每课秒数，默认 20]
#
# 前置: OrcaLab 已运行（默认 localhost:50051）、资产包已导入。
# 原理: 每课驻留到时后发 SIGINT，课程自带的 KeyboardInterrupt 分支会
#       优雅退出（scene.close()），不会残留场景连接。
# 特例: 第 13 课自动带 --default-scene（批跑模式下不等待手动拖拽），
#       且步进完成后自行退出，超时仅作兜底。
# 中断: 观察中途想停全流程，连按 Ctrl+C 两次。

set -u

DUR="${1:-20}"
ADDR="${ORCAGYM_ADDR:-localhost:50051}"
PY="${ORCA_PYTHON:-python}"

LESSONS=(
  examples.euler.beginner.stage1_scene_basics.lesson_01_hello_world.run
  examples.euler.beginner.stage1_scene_basics.lesson_02_load_object.run
  examples.euler.beginner.stage1_scene_basics.lesson_03_load_robot.run
  examples.euler.beginner.stage1_scene_basics.lesson_04_load_scene.run
  examples.euler.beginner.stage1_scene_basics.lesson_05_compose_scene.run
  examples.euler.beginner.stage1_scene_basics.lesson_06_identify_object.run
  examples.euler.beginner.stage2_scene_editing.lesson_07_move_object.run
  examples.euler.beginner.stage2_scene_editing.lesson_08_rotate_object.run
  examples.euler.beginner.stage2_scene_editing.lesson_09_scale_object.run
  examples.euler.beginner.stage2_scene_editing.lesson_10_color_object.run
  examples.euler.beginner.stage2_scene_editing.lesson_11_duplicate_objects.run
  examples.euler.beginner.stage2_scene_editing.lesson_12_delete_object.run
  examples.euler.beginner.stage3_simulation_time.lesson_13_step_simulation.run
)

FAIL=0
for lesson in "${LESSONS[@]}"; do
  num="$(echo "$lesson" | grep -o 'lesson_[0-9]*' | grep -o '[0-9]*')"
  extra_args=()
  if [ "$num" = "13" ]; then
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
  echo "全部 13 课跑完，无异常退出"
else
  echo "跑完，但有 ${FAIL} 课异常退出，请回看上方日志"
fi
echo "=================================================================="
