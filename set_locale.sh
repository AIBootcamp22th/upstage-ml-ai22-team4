#!/bin/bash
# UTF-8 로케일 설정 스크립트

# 시스템에 설치된 C.UTF-8 로케일 사용
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

echo "로케일이 설정되었습니다:"
locale | grep -E "LANG|LC_ALL"

echo ""
echo "이제 편집기를 실행할 수 있습니다:"
echo "  vi 파일명"
echo "  또는"
echo "  ./code/edit_utf8.sh 파일명"
