#!/bin/bash
# vi 편집기 UTF-8 테스트

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

echo "로케일 설정 확인:"
locale | grep -E "LANG|LC_ALL"

echo ""
echo "파일 인코딩 확인:"
file -bi train_model.py

echo ""
echo "한글 샘플 확인:"
head -3 train_model.py | tail -1

echo ""
echo "vi 편집기 실행 방법:"
echo "  ./edit_utf8.sh train_model.py"
echo "  또는"
echo "  export LANG=C.UTF-8 && export LC_ALL=C.UTF-8 && vi train_model.py"
