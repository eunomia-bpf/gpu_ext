#!/bin/bash
# Doxygen 运行脚本

cd "$(dirname "$0")"

echo "=== Checking if doxygen is installed ==="
if ! command -v doxygen &> /dev/null; then
    echo "ERROR: doxygen is not installed"
    echo "Please run: sudo apt-get install doxygen graphviz"
    exit 1
fi

echo "doxygen version: $(doxygen --version)"

if ! command -v dot &> /dev/null; then
    echo "WARNING: graphviz (dot) is not installed"
    echo "Call graphs will not be generated"
    echo "Please run: sudo apt-get install graphviz"
fi

echo ""
echo "=== Running doxygen ==="
doxygen Doxyfile.uvm

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Success! ==="
    echo "Output directory: doxygen_output/html/"
    echo ""
    echo "=== Open with browser: ==="
    echo "firefox doxygen_output/html/index.html"
    echo "google-chrome doxygen_output/html/index.html"
    echo ""
    echo "=== Or start a web server: ==="
    echo "cd doxygen_output/html && python3 -m http.server 8080"
    echo "Then visit: http://localhost:8080"
    echo ""
    echo "=== Check for warnings: ==="
    echo "less doxygen_output/doxygen_warnings.txt"

    # 尝试自动打开浏览器
    if command -v xdg-open &> /dev/null; then
        echo ""
        read -p "Open in browser now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            xdg-open doxygen_output/html/index.html
        fi
    fi
else
    echo "ERROR: doxygen failed"
    echo "Check doxygen_output/doxygen_warnings.txt for errors"
fi
