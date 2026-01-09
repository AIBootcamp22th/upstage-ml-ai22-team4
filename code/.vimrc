" UTF-8 인코딩 설정
set encoding=utf-8
set fileencoding=utf-8
set fileencodings=utf-8,cp949,euc-kr,ucs-bom,latin1

" 한글 입력 지원
set langmap=ㅁa,ㅠb,ㅊc,ㅇd,ㄷe,ㄹf,ㅎg,ㅗh,ㅑi,ㅓj,ㅏk,ㅣl,ㅡm,ㅜn,ㅐo,ㅔp,ㅂq,ㄱr,ㄴs,ㅅt,ㅕu,ㅍv,ㅈw,ㅌx,ㅛy,ㅋz
set iminsert=0
set imsearch=0

" 파일 저장 시 UTF-8로 저장
autocmd BufWritePre * setlocal fileencoding=utf-8

" Python 파일에 대한 특별 설정
autocmd FileType python setlocal fileencoding=utf-8 encoding=utf-8
