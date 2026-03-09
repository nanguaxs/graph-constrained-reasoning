docker run --gpus all -d --name gcr \
  -p 2026:22 \
  -v /mnt/c/Users/liangchuan/work/graph-constrained-reasoning:/workspace/graph-constrained-reasoning \
  -v /mnt/c/Users/liangchuan/work/graph-constrained-reasoning/.vscode-server:/root/.vscode-server \
  gcr:latest \
  /bin/bash -c "service ssh start && tail -f /dev/null"
