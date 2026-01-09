#!/bin/bash
# UTF-8 인코딩으로 vi 편집기 실행 (간단한 버전)

# C.UTF-8 로케일 사용 (시스템에 기본 설치됨)
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# vi 실행 시 UTF-8 설정
if [ -f ~/.vimrc ]; then
    vim -u ~/.vimrc "$@"
elif [ -f /data/ephemeral/home/py310/.vimrc ]; then
    vim -u /data/ephemeral/home/py310/.vimrc "$@"
else
    # vi 내에서 UTF-8 설정
    vim -c "set encoding=utf-8" -c "set fileencoding=utf-8" "$@"
fi
