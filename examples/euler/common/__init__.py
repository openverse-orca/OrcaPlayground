"""OrcaPlayground Euler 示例公共库。

集中存放被多个示例（Lesson 05–10 等）复制/衍生的公共环境代码，作为唯一来源：
- g1_base_env.py（Phase B 迁入）
- g1_locomotion.py
- online_verifier.py
- scene_scanner.py
- simple_env.py（Phase E 迁入）

通行约定：示例脚本在文件头执行
``sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`` 后以
``from common.xxx import ...`` 方式导入，不设置 pip install。
"""