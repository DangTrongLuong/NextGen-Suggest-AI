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
                        echo "Phát hiện file .env thật tại $REAL_ENV, đang nạp vào workspace..."
                        cp "$REAL_ENV" .env
                    else
                        echo "Thử nạp file .env từ host qua Docker Daemon..."
                        docker run --rm -v "$REAL_ENV":/host_env alpine cat /host_env > .env || true
                    fi

                    if [ ! -s .env ] && [ -f .env.example ]; then
                        echo "Không nạp được .env thật, dùng .env.example..."
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

        stage('Deploy Docker Container (Zero Downtime CD)') {
            steps {
                script {
                    echo '=== Bắt đầu Blue-Green Deployment (0ms Downtime) ==='
                    sh '''
                        set -eux

                        REAL_ENV="/www/wwwroot/ai.datn-nextgen-suggest.site/NextGen-Suggest-AI/.env"
                        if [ -f "$REAL_ENV" ]; then
                            cp "$REAL_ENV" .env
                        else
                            docker run --rm -v "$REAL_ENV":/host_env alpine cat /host_env > .env || true
                        fi

                        if [ -d "nginx-ai.conf" ]; then
                            rm -rf nginx-ai.conf
                        fi

                        docker network create nextgen-network || true

                        IS_BLUE_RUNNING=$(docker ps --filter "status=running" --format '{{.Names}}' | grep -w "ai-blue" || true)

                        if [ -n "$IS_BLUE_RUNNING" ]; then
                            TARGET="ai-green"
                            OLD="ai-blue"
                        else
                            TARGET="ai-blue"
                            OLD="ai-green"
                        fi

                        echo "Target: $TARGET | Old: $OLD"

                        docker compose build $TARGET
                        docker compose up -d $TARGET

                        HEALTHY=0
                        for i in $(seq 1 60); do
                            STATUS_CODE=$(docker exec $TARGET curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "000")
                            if [ "$STATUS_CODE" = "200" ]; then
                                echo "$TARGET is healthy!"
                                HEALTHY=1
                                break
                            fi
                            sleep 3
                        done

                        if [ "$HEALTHY" -ne 1 ]; then
                            echo "$TARGET healthcheck failed. Keeping $OLD."
                            docker compose stop $TARGET || true
                            exit 1
                        fi

                        docker compose up -d ai

                        docker exec -i nextgen-ai sh -c "cat << 'EOF' > /etc/nginx/conf.d/default.conf
upstream fastapi_cluster {
    server ${TARGET}:8000;
}

server {
    listen 8000;
    server_name _;

    client_max_body_size 50M;

    location / {
        proxy_pass http://fastapi_cluster;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF"

                        docker exec nextgen-ai nginx -s reload || true
                        docker compose stop $OLD || true
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
            sh '''
                echo "Đang dọn dẹp Docker build cache và image rác..."
                docker builder prune -a -f || true
                docker image prune -a -f || true
            '''
            echo 'Đang dọn dẹp Jenkins workspace...'
            cleanWs()
        }
    }
}
