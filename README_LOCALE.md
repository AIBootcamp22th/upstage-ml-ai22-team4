# 로케일 및 편집기 설정 가이드

## 문제
- `ko_KR.UTF-8` 로케일이 시스템에 설치되어 있지 않음
- `nano` 편집기가 설치되어 있지 않음

## 해결 방법

### 1. 올바른 로케일 설정

시스템에 `ko_KR.UTF-8`이 없으므로 `C.UTF-8`을 사용하세요:

```bash
# 방법 1: 직접 설정
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# 방법 2: 스크립트 사용
source set_locale.sh
```

### 2. 편집기 사용

#### vi/vim 사용 (권장)
```bash
# UTF-8 설정 후 vi 실행
export LANG=C.UTF-8 && export LC_ALL=C.UTF-8 && vi 파일명

# 또는 편집 스크립트 사용
./code/edit_utf8.sh 파일명
```

#### vi에서 UTF-8 설정
vi 내에서:
```
:set encoding=utf-8
:set fileencoding=utf-8
:e
```

### 3. 영구적으로 환경 변수 설정 (선택사항)

`~/.bashrc` 또는 `~/.bash_profile`에 추가:
```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
```

그 다음:
```bash
source ~/.bashrc
```

### 4. nano 설치 (선택사항)

nano를 사용하고 싶다면:
```bash
# Ubuntu/Debian
apt-get update && apt-get install -y nano

# CentOS/RHEL
yum install -y nano
```

## 확인

로케일이 올바르게 설정되었는지 확인:
```bash
locale
```

출력 예시:
```
LANG=C.UTF-8
LC_ALL=C.UTF-8
...
```

## 참고

- `C.UTF-8`은 UTF-8 인코딩을 완전히 지원합니다
- 한글, 일본어, 중국어 등 모든 유니코드 문자를 처리할 수 있습니다
- `ko_KR.UTF-8`이 없어도 문제없이 한글을 사용할 수 있습니다
