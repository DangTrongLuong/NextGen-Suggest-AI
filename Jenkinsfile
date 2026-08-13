pipeline {
    agent any

    environment {
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
        PYTHONDONTWRITEBYTECODE = '1'
        PYTHONUNBUFFERED = '1'
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 25, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    stages {
        stage('Checkout Source') {
            steps {
                echo 'Đang checkout mã nguồn FastAPI...'
                checkout scm
            }
        }

        stage('Show Environment') {
            steps {
                sh '''
                    set -eux
                    python3 --version || true
                    docker --version
                    docker compose version
                    pwd
                '''
            }
        }

        stage('Validate Project') {
            steps {
                script {
                    if (fileExists('requirements.txt')) {
                        echo 'Tìm thấy requirements.txt'
                    } else {
                        error 'Không tìm thấy requirements.txt ở thư mục gốc.'
                    }
                }
            }
        }

        stage('Prepare Environment') {
            steps {
                sh '''
                    set -eux
                    REAL_ENV="/www/wwwroot/ai.datn-nextgen-suggest.site/NextGen-Suggest-AI/.env"
                    if [ -f "$REAL_ENV" ]; then
                        echo "Phát hiện file .env thật tại $REAL_ENV, đang nạp vào build workspace..."
                        cp "$REAL_ENV" .env
                    elif [ ! -f .env ] && [ -f .env.example ]; then
                        echo "Tự động sao chép .env.example -> .env cho môi trường CI..."
                        cp .env.example .env
                    fi
                '''
            }
        }

        stage('Syntax Check') {
            steps {
                sh '''
                    set -eux
                    echo "=== Kiểm tra cú pháp Python nhanh ==="
                    if [ -d app ]; then python3 -m compileall app; fi
                    if [ -d scripts ]; then python3 -m compileall scripts; fi
                    if [ -f main.py ]; then python3 -m py_compile main.py; fi
                '''
            }
        }

        stage('Deploy Docker Container (CD)') {
            steps {
                script {
                    echo '=== Bắt đầu quy trình CD: Build Docker Image & Deploy bằng Docker Compose ==='
                    sh '''
                        set -eux

                        REAL_ENV="/www/wwwroot/ai.datn-nextgen-suggest.site/NextGen-Suggest-AI/.env"
                        if [ -f "$REAL_ENV" ]; then
                            echo "Đảm bảo workspace có file .env mới nhất từ $REAL_ENV..."
                            cp "$REAL_ENV" .env
                        fi

                        # 1. Đảm bảo network chung nextgen-network tồn tại
                        docker network create nextgen-network || true

                        # 2. Tái tạo container bằng docker-compose.yml của repo
                        if [ -f "$REAL_ENV" ]; then
                            docker compose --env-file "$REAL_ENV" up -d --build ai
                        else
                            docker compose up -d --build ai
                        fi

                        # 3. Kiểm tra trạng thái Container vừa khởi chạy
                        sleep 5
                        docker ps | grep nextgen-ai || true

                        # 4. Tự động xóa Build Cache và Image rác dư thừa sau khi build xong để chống đầy ổ đĩa
                        docker builder prune -f || true
                        docker image prune -f || true
                    '''
                }
            }
        }
    }

    post {
        success {
            echo 'KẾT QUẢ: FastAPI CI/CD pipeline thành công.'
        }

        failure {
            echo 'KẾT QUẢ: Pipeline thất bại. Hãy kiểm tra Console Output.'
        }

        always {
            echo "Trạng thái cuối cùng: ${currentBuild.currentResult}"
            echo 'Đang dọn dẹp Jenkins workspace...'
            cleanWs()
        }
    }
}
