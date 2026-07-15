pipeline {
    agent any

    environment {
        VENV_DIR = '.venv-jenkins'
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
        PYTHONDONTWRITEBYTECODE = '1'
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
                echo 'Đang thực hiện checkout mã nguồn FastAPI từ Git...'
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
                        echo "Xác thực thành công: Tìm thấy requirements.txt"
                    } else if (fileExists('pyproject.toml')) {
                        echo "Xác thực thành công: Tìm thấy pyproject.toml"
                    } else {
                        error "LỖI: Không tìm thấy file requirements.txt hoặc pyproject.toml ở thư mục gốc của dự án. Hủy pipeline!"
                    }
                }
            }
        }

        stage('Create Virtual Environment') {
            steps {
                sh '''
                    set -eux
                    python3 -m venv $VENV_DIR
                '''
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    set -eux
                    . $VENV_DIR/bin/activate
                    pip install --upgrade pip
                    
                    if [ -f "requirements.txt" ]; then
                        echo "Đang cài đặt thư viện từ requirements.txt..."
                        pip install -r requirements.txt
                    elif [ -f "pyproject.toml" ]; then
                        echo "Đang cài đặt thư viện từ pyproject.toml..."
                        if command -v poetry >/dev/null 2>&1; then
                            poetry install
                        else
                            pip install .
                        fi
                    fi
                    
                    if ! pip show pytest >/dev/null 2>&1; then
                        echo "Cài đặt bổ sung pytest phục vụ kiểm thử CI..."
                        pip install pytest
                    fi
                '''
            }
        }

        stage('Syntax Check') {
            steps {
                sh '''
                    set -eux
                    . $VENV_DIR/bin/activate
                    
                    echo "=== Kiểm tra cú pháp (Syntax Check) ==="
                    python -m compileall -x '(\\.git|\\.venv|\\.venv-jenkins|venv|__pycache__|logs|cache|models|scratch|data)' .
                    
                    echo "=== Kiểm tra import module FastAPI chính ==="
                    if [ -f "app/main.py" ]; then
                        PYTHONPATH=. python -c "import app.main; print('Import app.main thành công!')"
                    elif [ -f "main.py" ]; then
                        PYTHONPATH=. python -c "import main; print('Import main thành công!')"
                    elif [ -f "src/main.py" ]; then
                        PYTHONPATH=. python -c "import src.main; print('Import src.main thành công!')"
                    else
                        echo "Không tìm thấy file entry point chính. Bỏ qua kiểm tra import."
                    fi
                '''
            }
        }

        stage('Test') {
            steps {
                sh '''
                    set -eux
                    . $VENV_DIR/bin/activate
                    
                    mkdir -p reports
                    
                    if python -m pytest --version >/dev/null 2>&1; then
                        echo "Bắt đầu chạy Unit Tests với pytest..."
                        python -m pytest --junitxml=reports/pytest.xml || echo "Có một số test bị thất bại nhưng tiếp tục để lưu báo cáo"
                    else
                        echo "CẢNH BÁO: Không tìm thấy công cụ pytest để chạy kiểm thử."
                    fi
                '''
            }
        }

        stage('Package Artifact') {
            steps {
                sh '''
                    set -eux
                    echo "=== Đóng gói source code thành .tar.gz ==="
                    tar --exclude='.git' \
                        --exclude='.venv' \
                        --exclude='.venv-jenkins' \
                        --exclude='venv' \
                        --exclude='__pycache__' \
                        --exclude='.env' \
                        --exclude='logs' \
                        --exclude='cache' \
                        --exclude='models' \
                        --exclude='scratch' \
                        --exclude='data' \
                        -czf fastapi-app.tar.gz .
                '''
                
                archiveArtifacts artifacts: 'fastapi-app.tar.gz'
                
                script {
                    if (fileExists('reports/pytest.xml')) {
                        archiveArtifacts artifacts: 'reports/pytest.xml'
                    }
                }
            }
        }
    }

    post {
        success {
            echo 'KẾT QUẢ: Pipeline chạy thành công tốt đẹp!'
        }
        unstable {
            echo 'KẾT QUẢ: Pipeline chạy hoàn thành nhưng trạng thái không ổn định (unstable)!'
        }
        failure {
            echo 'KẾT QUẢ: Pipeline thất bại! Vui lòng kiểm tra lại log chi tiết.'
        }
        always {
            script {
                if (fileExists('reports/pytest.xml')) {
                    echo "Publishing test results..."
                    junit testResults: 'reports/pytest.xml', allowEmptyResults: true
                }
            }
            
            echo 'Đang dọn dẹp thư mục làm việc (Workspace)...'
            cleanWs()
        }
    }
}
