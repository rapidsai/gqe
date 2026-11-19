ARG IMAGE

FROM $IMAGE

# initialize-devcontainer.sh records the host IDs before this image is built.
COPY devcontainer-user-ids /tmp/devcontainer-user-ids

# Ensure a group named 'hardware' exists with GID=$DEV_GID; if a group with $DEV_GID exists but is not named 'hardware', rename it
RUN . /tmp/devcontainer-user-ids && \
    existing_group=$(getent group $DEV_GID | head -n1 | cut -d: -f1) && \
    if [ -z "$existing_group" ]; then \
        groupadd --gid $DEV_GID hardware; \
    elif [ "$existing_group" != "hardware" ]; then \
        groupmod -n hardware "$existing_group"; \
    fi

# Create the gqe-dev user with the specified DEV_UID and DEV_GID (hardware)
RUN . /tmp/devcontainer-user-ids && \
    echo "DEV_UID: $DEV_UID, DEV_GID: $DEV_GID" && \
    useradd --uid $DEV_UID --gid hardware -m gqe-dev && \
    unlink /tmp/devcontainer-user-ids

# Give conda a user-owned package cache and make env directories writable.
RUN mkdir -p /home/gqe-dev/.conda/pkgs && \
    chown -R gqe-dev:hardware /home/gqe-dev/.conda && \
    find /conda/envs/gqe -type d -exec chown gqe-dev:hardware {} + && \
    chown -R gqe-dev:hardware /conda/envs/gqe/conda-meta
ENV CONDA_PKGS_DIRS=/home/gqe-dev/.conda/pkgs

# Add gqe-dev to sudoers
RUN mkdir -p /etc/sudoers.d && \
    echo "gqe-dev ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/gqe-dev && \
    chmod 0440 /etc/sudoers.d/gqe-dev
