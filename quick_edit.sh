#!/bin/bash
# 빠른 편집 스크립트

# UTF-8 로케일 설정
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

if [ $# -eq 0 ]; then
    echo "사용법: ./quick_edit.sh <파일명>"
    echo "예시: ./quick_edit.sh code/train_model.py"
    exit 1
fi

# vi 실행 (UTF-8 설정 포함)
if [ -f ~/.vimrc ]; then
    vim -u ~/.vimrc "$@"
elif [ -f /data/ephemeral/home/py310/.vimrc ]; then
    vim -u /data/ephemeral/home/py310/.vimrc "$@"
else
    vim -c "set encoding=utf-8" -c "set fileencoding=utf-8" "$@"
fi
