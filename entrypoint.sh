#!/bin/bash
# Entrypoint script to ensure a user with the Docker-specified UID exists
 
# Use environment variables passed to the docker run command to set the username, uid and guid
# IMPORTANT NOTE: the uid and guid should be passed in as environment variables
# IMPORTANT NOTE: not via the --user flag in docker run, using --user will result
# IMPORTANT NOTE: in this script not being run as the root user, which is necessary for setting up sudo
USERNAME=${HOST_USERNAME:-dev_user}
USER_UID=${HOST_USER_UID:-1000}
USER_GID=${HOST_USER_GID:-1000}
 
# Create a new group with the USER_GID if it does not already exist
if ! getent group $USER_GID > /dev/null; then
    groupadd -g $USER_GID $USERNAME
fi
 
# Create a new user with the USER_UID and USER_GID if it does not already exist
if ! getent passwd $USER_UID > /dev/null; then
    useradd -l -u $USER_UID -g $USER_GID -m $USERNAME
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi

# Get the actual username for this UID (in case user already existed with different name)
ACTUAL_USERNAME=$(getent passwd $USER_UID | cut -d: -f1)

# Get the actual group name for this GID
GROUP_NAME=$(getent group $USER_GID | cut -d: -f1)

# Try to fix conda permissions first (works in most cases)
if chown -R $USER_UID:$USER_GID /conda 2>/dev/null; then
    echo "Using system conda environment"
    # Set up conda for the user
    mkdir -p /home/$ACTUAL_USERNAME/.conda
    echo 'export PATH="/conda/bin:$PATH"' >> /home/$ACTUAL_USERNAME/.bashrc
else
    echo "Creating user-specific conda environment"
    # Create user's conda environment directory
    mkdir -p /home/$ACTUAL_USERNAME/.conda/envs/gqe
    
    # Pack and unpack the conda environment for the new user
    # Note: conda-pack is installed in the gqe environment
    if /conda/envs/gqe/bin/conda-pack -n gqe -o /tmp/gqe.tar.gz 2>/dev/null; then
        tar -xzf /tmp/gqe.tar.gz -C /home/$ACTUAL_USERNAME/.conda/envs/gqe
        rm /tmp/gqe.tar.gz
        
        # Set up conda for the user
        echo 'export PATH="/home/'$ACTUAL_USERNAME'/.conda/envs/gqe/bin:$PATH"' >> /home/$ACTUAL_USERNAME/.bashrc
        
        # Fix permissions
        chown -R $USER_UID:$USER_GID /home/$ACTUAL_USERNAME/.conda
    else
        echo "Warning: conda-pack not available, linking to system conda instead"
        echo 'export PATH="/conda/bin:$PATH"' >> /home/$ACTUAL_USERNAME/.bashrc        
    fi
fi

echo 'source activate gqe' >> /home/$ACTUAL_USERNAME/.bashrc

# Ensure proper ownership of user's home directory
chown -R $USER_UID:$USER_GID /home/$ACTUAL_USERNAME

# Switch to the created user and execute the command
exec gosu $ACTUAL_USERNAME "$@"