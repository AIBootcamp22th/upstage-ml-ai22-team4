#!/bin/bash
# UTF-8 인코딩으로 vi 편집기 실행 스크립트

# UTF-8 로케일 설정 (시스템에 설치된 로케일 사용)
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# vi 실행 (UTF-8 설정 포함)
if [ -f ~/.vimrc ]; then
    vim -u ~/.vimrc "$@"
elif [ -f /data/ephemeral/home/py310/.vimrc ]; then
    vim -u /data/ephemeral/home/py310/.vimrc "$@"
else
    vim -c "set encoding=utf-8" -c "set fileencoding=utf-8" "$@"
fi
