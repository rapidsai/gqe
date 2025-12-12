ARG IMAGE

FROM $IMAGE

ARG DEV_UID
ARG DEV_GID

# Ensure a group named 'hardware' exists with GID=$DEV_GID; if a group with $DEV_GID exists but is not named 'hardware', rename it
RUN existing_group=$(getent group $DEV_GID | head -n1 | cut -d: -f1) && \
    if [ -z "$existing_group" ]; then \
        groupadd --gid $DEV_GID hardware; \
    elif [ "$existing_group" != "hardware" ]; then \
        groupmod -n hardware "$existing_group"; \
    fi

# Create the gqe-dev user with the specified DEV_UID and DEV_GID (hardware)
RUN echo "DEV_UID: $DEV_UID, DEV_GID: $DEV_GID" && \
    useradd --uid $DEV_UID --gid hardware -m gqe-dev

# Add gqe-dev to sudoers
RUN mkdir -p /etc/sudoers.d && \
    echo "gqe-dev ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/gqe-dev && \
    chmod 0440 /etc/sudoers.d/gqe-dev
