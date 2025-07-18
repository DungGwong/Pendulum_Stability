#!/bin/bash

# 定义 Python 解释器路径（根据实际环境调整，如 python3、python3.11 等）
PYTHON_CMD="python3"

# 检查 Python 解释器是否可用
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "错误：未找到 Python 解释器 $PYTHON_CMD，请检查环境配置。"
    exit 1
fi

# 第一步：运行 Train_Result.py
echo "开始运行 Train_Result.py..."
$PYTHON_CMD Train_Result.py

# 检查上一步是否成功执行（$? 为上一条命令的退出码，0 表示成功）
if [ $? -ne 0 ]; then
    echo "错误：Train_Result.py 执行失败，终止流程。"
    exit 1
fi

# 第二步：运行 Test_Result.py（仅当前一步成功时执行）
echo "Train_Result.py 执行成功，开始运行 Test_Result.py..."
$PYTHON_CMD Test_Result.py

# 检查 Test_Result.py 执行结果
if [ $? -ne 0 ]; then
    echo "错误：Test_Result.py 执行失败。"
    exit 1
fi

# 所有步骤完成
echo "所有脚本执行成功！"
exit 0