pipeline {
    agent any

    environment {
        VENV_DIR = '.venv-jenkins'
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

                    python3 --version
                    pip3 --version
                    git --version
                    pwd
                '''
            }
        }

        stage('Validate Project') {
            steps {
                script {
                    if (fileExists('requirements.txt')) {
                        echo 'Tìm thấy requirements.txt'
                    } else if (fileExists('pyproject.toml')) {
                        echo 'Tìm thấy pyproject.toml'
                    } else {
                        error '''
Không tìm thấy requirements.txt hoặc pyproject.toml
ở thư mục gốc của repository.
'''
                    }

                    if (fileExists('app/main.py')) {
                        echo 'Phát hiện FastAPI entry point: app/main.py'
                    } else if (fileExists('main.py')) {
                        echo 'Phát hiện FastAPI entry point: main.py'
                    } else if (fileExists('src/main.py')) {
                        echo 'Phát hiện FastAPI entry point: src/main.py'
                    }
                }
            }
        }

        stage('Prepare Environment') {
            steps {
                script {
                    def realEnvPath = '/www/wwwroot/ai.datn-nextgen-suggest.site/NextGen-Suggest-AI/.env'
                    if (fileExists(realEnvPath)) {
                        echo "Phát hiện file .env thật tại ${realEnvPath}, đang nạp vào build workspace..."
                        sh "cp ${realEnvPath} .env"
                    } else if (!fileExists('.env') && fileExists('.env.example')) {
                        echo 'Tự động sao chép .env.example -> .env cho môi trường CI...'
                        sh 'cp .env.example .env'
                    }
                }
            }
        }

        stage('Create Virtual Environment') {
            steps {
                sh '''
                    set -eux

                    rm -rf "$VENV_DIR"
                    python3 -m venv "$VENV_DIR"
                '''
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    set -eux

                    . "$VENV_DIR/bin/activate"

                    python -m pip install --upgrade pip setuptools wheel

                    if [ -f requirements.txt ]; then
                        echo "Đang cài dependency từ requirements.txt..."
                        python -m pip install --no-cache-dir -r requirements.txt
                    elif [ -f pyproject.toml ]; then
                        echo "Đang cài dependency từ pyproject.toml..."

                        if command -v poetry >/dev/null 2>&1; then
                            poetry install --no-interaction
                        else
                            python -m pip install --no-cache-dir .
                        fi
                    fi

                    if ! python -m pytest --version >/dev/null 2>&1; then
                        echo "Cài pytest phục vụ CI..."
                        python -m pip install --no-cache-dir pytest
                    fi
                '''
            }
        }

        stage('Syntax Check') {
            steps {
                sh '''
                    set -eux

                    . "$VENV_DIR/bin/activate"

                    echo "=== Kiểm tra cú pháp Python ==="

                    if [ -d app ]; then
                        python -m compileall app
                    fi

                    if [ -d scripts ]; then
                        python -m compileall scripts
                    fi

                    if [ -f main.py ]; then
                        python -m py_compile main.py
                    fi

                    if [ -d src ]; then
                        python -m compileall src
                    fi

                    echo "Kiểm tra cú pháp hoàn tất."
                '''
            }
        }

        stage('Test') {
            steps {
                sh '''
                    set -eux

                    . "$VENV_DIR/bin/activate"
                    mkdir -p reports

                    TEST_FILES=$(find . \
                        -path "./$VENV_DIR" -prune -o \
                        -path "./.git" -prune -o \
                        -path "./venv" -prune -o \
                        -type f \
                        \\( -name "test_*.py" -o -name "*_test.py" \\) \
                        -print | head -n 1)

                    if [ -n "$TEST_FILES" ]; then
                        echo "Phát hiện test, bắt đầu chạy pytest..."

                        python -m pytest \
                            --junitxml=reports/pytest.xml
                    else
                        echo "Không tìm thấy test Python. Bỏ qua pytest."
                    fi
                '''
            }
            post {
                always {
                    script {
                        if (fileExists('reports/pytest.xml')) {
                            junit(
                                testResults: 'reports/pytest.xml',
                                allowEmptyResults: true
                            )
                        }
                    }
                }
            }
        }

        stage('Package Artifact') {
            steps {
                sh '''
                    set -eux
                    echo "=== Đóng gói source code FastAPI ==="
                    tar --exclude='.git' --exclude='.venv' --exclude='venv' --exclude='__pycache__' --exclude='models' -czf fastapi-app.tar.gz app/ scripts/ requirements.txt Dockerfile docker-compose.yml .env.example || true
                    ls -lh fastapi-app.tar.gz || true
                '''

                archiveArtifacts(
                    artifacts: 'fastapi-app.tar.gz',
                    fingerprint: true,
                    allowEmptyArchive: true
                )

                script {
                    if (fileExists('reports/pytest.xml')) {
                        archiveArtifacts(
                            artifacts: 'reports/pytest.xml',
                            allowEmptyArchive: true
                        )
                    }
                }
            }
        }

        stage('Deploy Docker Container (CD)') {
            steps {
                script {
                    echo '=== Bắt đầu quy trình CD: Build Docker Image & Deploy bằng Docker Compose ==='
                    sh '''
                        set -eux

                        # 1. Đảm bảo network chung nextgen-network tồn tại
                        docker network create nextgen-network || true

                        # 2. Tái tạo container bằng docker-compose.yml của repo
                        docker compose up -d --build ai

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
            echo 'KẾT QUẢ: FastAPI CI pipeline thành công.'
        }

        unstable {
            echo 'KẾT QUẢ: Pipeline hoàn tất nhưng có kiểm thử không ổn định.'
        }

        failure {
            echo 'KẾT QUẢ: Pipeline thất bại. Hãy kiểm tra stage bị lỗi trong Console Output.'
        }

        always {
            echo "Trạng thái cuối cùng: ${currentBuild.currentResult}"
            echo 'Đang dọn dẹp Jenkins workspace...'
            cleanWs()
        }
    }
}
