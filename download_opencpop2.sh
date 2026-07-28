#!/bin/bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

COOKIE_FILE=/tmp/opencpop_cookies.txt
touch "$COOKIE_FILE"
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BASE_URL="https://wenet.org.cn/downloads?dataset=opencpop"

echo "=== 1. 访问 TERMS_OF_ACCESS 获取 cookie ==="
curl -kL -c "$COOKIE_FILE" -A "$UA" -s -D- "${BASE_URL}&name=TERMS_OF_ACCESS" -o /dev/null | head -20
echo ""

echo "=== 2. cookies 内容 ==="
cat "$COOKIE_FILE" 2>/dev/null
echo ""

echo "=== 3. 测试下载文件大小 ==="
for name in midis textgrids wavs segments; do
    echo "--- $name ---"
    curl -kL -b "$COOKIE_FILE" -A "$UA" -s -D- "${BASE_URL}&name=${name}.zip" -o /tmp/${name}.zip | head -5
    ls -lh /tmp/${name}.zip 2>/dev/null
    file /tmp/${name}.zip 2>/dev/null
    echo ""
done
