#!/bin/bash

# Add conda activation to .bashrc
echo "source /conda/bin/activate gqe" >> /home/gqe-dev/.bashrc
echo '
if [ -L /tmp/ssh-agent.socket ] && [ -S /tmp/ssh-agent.socket ]; then
    export SSH_AUTH_SOCK=/tmp/ssh-agent.socket
fi
' >> /home/gqe-dev/.bashrc
