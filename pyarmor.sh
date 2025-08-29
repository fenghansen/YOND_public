#!/bin/bash

# 定义变量
PY_FILE="isp_algos.py"
PRIVATE_DIR="private"
PYARMOR_VERSION="7.7.4"

# 检查源文件是否存在
if [ ! -f "$PY_FILE" ]; then
    echo "错误：文件 $PY_FILE 不存在！"
    exit 1
fi

# 检查pyarmor是否安装及版本是否正确
if ! command -v pyarmor &> /dev/null; then
    echo "错误：pyarmor 未安装！请先安装 pyarmor $PYARMOR_VERSION"
    exit 1
fi

INSTALLED_VERSION=$(pyarmor --version | awk '{print $2}')
if [ "$INSTALLED_VERSION" != "$PYARMOR_VERSION" ]; then
    echo "警告：pyarmor 版本不是 $PYARMOR_VERSION"
    echo "已安装版本：$INSTALLED_VERSION"
    echo "继续执行加密操作..."
    # 若希望严格检查版本，可取消下面一行的注释
    # exit 1
fi

# 创建private目录（如果不存在）
mkdir -p "$PRIVATE_DIR"

# 使用pyarmor加密文件
echo "开始使用pyarmor加密 $PY_FILE..."
pyarmor obfuscate --exact "$PY_FILE"

# 检查加密是否成功
if [ $? -ne 0 ]; then
    echo "错误：pyarmor加密失败！"
    exit 1
fi

# 移动源文件到private目录
echo "将源文件移动到 $PRIVATE_DIR 目录..."
mv "$PY_FILE" "$PRIVATE_DIR/"

# 检查移动是否成功
if [ $? -ne 0 ]; then
    echo "错误：移动源文件失败！"
    exit 1
fi

echo "操作完成！"
echo "加密后的文件已生成在 dist 目录中"
echo "源文件已保存到 $PRIVATE_DIR/$PY_FILE"
