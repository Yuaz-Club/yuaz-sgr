#!/bin/bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

DATA_DIR="/root/autodl-tmp/opencpop"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BASE_URL="https://wenet.org.cn/downloads?dataset=opencpop&name="

download_file() {
    local name="$1"
    local filename="${name}.zip"
    echo "=== 下载 $filename ==="
    curl -kL -A "$UA" -o "$filename" "${BASE_URL}${name}"
    if [ $? -eq 0 ] && [ -f "$filename" ]; then
        local size=$(stat -c%s "$filename" 2>/dev/null || stat -f%z "$filename" 2>/dev/null)
        echo "  完成: $filename ($size bytes)"
        return 0
    else
        echo "  ERROR: $filename 下载失败"
        return 1
    fi
}

# 下载所有文件
download_file "midis"
download_file "textgrids"
download_file "wavs"

echo ""
echo "=== 所有文件 ==="
ls -lh "$DATA_DIR"/*.zip 2>/dev/null
echo ""
echo "=== 磁盘使用 ==="
df -h /root/autodl-tmp | tail -1
