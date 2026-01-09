# vi 편집기에서 한글 깨짐 문제 해결 가이드

## 문제
vi 편집기에서 Python 파일을 열 때 한글이 깨져서 보이는 문제

## 원인
시스템 로케일이 POSIX로 설정되어 있어 UTF-8 파일을 제대로 표시하지 못함

## 해결 방법

### 방법 1: 편집 스크립트 사용 (가장 쉬운 방법)
```bash
cd /data/ephemeral/home/py310/code
./edit_utf8.sh train_model.py
# 또는
./vi_utf8.sh train_model.py
```

### 방법 2: 환경 변수 설정 후 vi 실행
```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
vi train_model.py
```

### 방법 3: vi에서 직접 설정
```bash
vi train_model.py
```
vi 내에서 다음 명령어 입력:
```
:set encoding=utf-8
:set fileencoding=utf-8
:e
```

### 방법 4: .vimrc 파일 사용
```bash
vi -u /data/ephemeral/home/py310/.vimrc train_model.py
```

### 방법 5: 영구적으로 환경 변수 설정 (선택사항)
`~/.bashrc` 또는 `~/.bash_profile`에 추가:
```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
```

## 참고사항
- 시스템에 `C.UTF-8` 로케일이 기본 설치되어 있습니다
- `ko_KR.UTF-8` 로케일은 설치되어 있지 않으므로 `C.UTF-8`을 사용하세요
- 모든 Python 파일은 UTF-8로 저장되어 있습니다
