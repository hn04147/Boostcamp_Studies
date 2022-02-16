## Docker
* ```docker pull "이미지 이름 : 태그"``` : 필요한 이미지 다운
  * ex ) ```docker pull mysql:8``` : mysql 8 버전의 이미지를 다운
* ```docker images``` : 다운받은 이미지 목록 확인
* ```docker run "이미지 이름 : 태그"``` : 이미지를 기반으로 컨테이너 생성
  * ex ) ```docker run --name mysql-tutorial -e MYSQL_ROOT_PASSWORD=1234 -d -p 3306:3306 mysql:8``` : 다운받은 MySQL 이미지 기반으로 Docker Container를 만들고 실행
    * ```--name mysql-tutorial``` : 컨테이너 이름, 지정하지 않으면 랜덤으로 생성
    * ```-e MYSQL_ROOT_PASSWORD=1234``` : 환경변수 설정
    * ```-d``` : 데몬(백그라운드) 모드. 컨테이너를 백그라운드 형태로 실행. 이 설정을 하지 않으면, 현재 실행하는 셀 위에서 컨테이너가 실행. 컨테이너의 로그를 바로 볼 수 있으나, 컨테이너를 나가면 실행 종료.
    * ```-p 3306:3306``` : 포트 지정. '로컬 호스트 포트 : 컨테이너 포트' 형태로, 로컬 포트 3306으로 접근 시 컨테이너 포트 3306으로 연결되도록 설정. MySQL은 기본적으로 3306 포트를 통해 통신 로컬 호스트 : 우리의 컴퓨터 컨테이너 : 컨테이너 이미지 내부
* ```docker ps``` : 실행중인 컨테이너 목록 확인
* ```docker exec -it "컨테이너 이름(ID)" /bin/bash``` : 컨테이너에 진입
* ```docker stop "컨테이너 이름(ID)"``` : 실행중인 컨테이너를 중지
* ```docker rm "컨테이너 이름(ID)"``` : 중지된 컨테이너 삭제

<br />

## MLFlow
* ```mlflow experiments create --experiment-name "experiment 이름"``` : experiment 생성
* ```mlflow experiments list``` : experiment 리스트 확인
