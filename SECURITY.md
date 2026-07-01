# Security

## Reporting Security Issues

> [!WARNING]
> Do not report security vulnerabilities through public GitHub issues!

Instead, please submit a private vulnerability report, see below.

## Reporting a Vulnerability

1. **NVIDIA Vulnerability Disclosure Program (preferred)**
   Submit through the NVIDIA Product Security Incident Response Team (PSIRT) web form (<https://www.nvidia.com/en-us/security/report-vulnerability/>)
   This is the fastest path to triage and tracking.

2. **Email NVIDIA PSIRT**
   `psirt@nvidia.com` — encrypt sensitive reports with the
   [NVIDIA PSIRT PGP key](https://www.nvidia.com/en-us/security/pgp-key).

3. **GitHub Private Vulnerability Reporting**
   Use the **Security and quality** tab on this repository → *Report a vulnerability*.

## Report Details

We prefer all communications to be in English.

Reports should include the following:

* reproducible example showing how the vulnerability can be exploited
* statement about the impact (including affected versions)

And we'd appreciate if they also include:

* statement about whether you are interested in implementing the fix yourself

## Disclosure Policy

NVIDIA PSIRT will acknowledge receipt and coordinate triage, fix development, and coordinated disclosure.

More on NVIDIA's response process: <https://www.nvidia.com/en-us/security/psirt-policies/>.

## Security Architecture & Context

GQE (GPU Query Engine) is a proof-of-concept SQL query engine that executes
analytics queries on NVIDIA GPUs. It is explicitly a research blueprint, not a
hardened production database.

**Software classification:** Server / Service (a long-running query service) with
an accompanying client SDK and CLI.

**Components and trust boundaries:**

- **Node manager (`gqe_node_manager`)** — exposes an Apache Arrow **Flight SQL**
  endpoint over plain gRPC/TCP (default `127.0.0.1:50051`). This is the primary
  network attack surface. It accepts SQL/Substrait query plans and DDL (`CREATE
  TABLE`, `COPY`, `CREATE EXTERNAL TABLE`), and returns Arrow record batches.
- **Task managers (`gqe_task_manager`)** — one subprocess per GPU, spawned by the
  node manager via `posix_spawn`. They expose an internal gRPC service (default
  base port `50051 + 1000`) and communicate across GPUs using NVSHMEM.
- **Rust client (`gqe-cli`)** — builds Substrait plans and drives the Flight SQL
  endpoint over `http://` (no TLS by default).
- **Storage layer** — reads Parquet via cuDF and a custom Parquet reader; ingested
  tables live in GPU memory for the lifetime of the node manager.

**Primary security responsibility:** GQE's interfaces (the Flight SQL endpoint, the
internal node↔task gRPC channels, and Parquet/Substrait parsing) currently provide
no authentication, authorization, or transport encryption. The intended security
boundary is the deployment environment: GQE assumes it runs on a trusted, isolated
host or cluster fabric with only trusted clients able to reach its ports.

## Threat Model

Threats below reference actual components. They are ordered by severity.

1. **Unauthenticated query execution and DDL on the Flight SQL endpoint.** The node
   manager's Flight SQL server (`src/node_manager/node_manager.cpp`,
   `src/node_manager/service.cpp`) is created with `FlightServerOptions` that set no
   `auth_handler` and no TLS, and the per-call `ServerCallContext` is ignored. Any
   party that can reach the listening port can run arbitrary SQL, create and drop
   tables, and load data — there is no notion of a user or permission.

2. **Server-side arbitrary file read via client-controlled paths.** `COPY ... FROM
   '<directory>'` and `CREATE EXTERNAL TABLE` pass client-supplied filesystem paths
   to the Parquet readers (`src/storage/parquet_reader.cpp`,
   `read_parquet_cudf`) with no path validation or sandboxing. A remote client can
   direct the server to read any Parquet-parseable file the server process can
   access, enabling information disclosure and path traversal on the host.

3. **Memory corruption from malformed Parquet input.** Attacker-supplied Parquet
   files are parsed in native/CUDA code by cuDF and by a hand-rolled custom reader
   (`include/gqe/storage/parquet_reader.hpp`). Crafted files can trigger
   memory-safety faults (crashes, potential corruption) in the parsing path.

4. **Untrusted plan deserialization (Substrait / physical plan / protobuf).** The
   service deserializes Substrait and physical plans from the wire
   (`service::execute_substrait_select`, `execute_physical_plan_select`,
   `src/logical/from_substrait.cpp`). Malformed or adversarial plans flow into the
   planner, optimizer, and executor, presenting a denial-of-service and
   memory-safety surface.

5. **Unauthenticated internal gRPC and inter-GPU traffic.** The node↔task channels
   and task migration use `grpc::InsecureChannelCredentials()`
   (`src/node_manager/spawn.cpp:161`, `src/rpc/task_migration.cpp:259`), and
   multi-GPU coordination uses NVSHMEM. If these ports/fabric are reachable beyond
   the trusted node, an attacker could inject plans or interfere with execution
   across processes.

6. **Resource exhaustion / denial of service.** Loaded tables reside in GPU memory
   with no per-client quota, and result sets are unbounded. A client can exhaust GPU
   or host memory. (A coarse `--query-timeout` exists but does not bound memory.)

7. **Privilege escalation / host compromise via the root container image.** The
   container image (`Dockerfile`) installs and runs `gqe_node_manager` and
   `gqe_task_manager` as root, with no `USER` directive. Combined with the
   server-side arbitrary file read above and any host bind mounts or
   privileged/`--device` runtime flags, a client that reaches the Flight SQL
   endpoint can read or modify host-visible files with root privileges, up to
   container escape on a misconfigured runtime.

8. **Disclosure or tampering of plaintext internal traffic.** Node↔task and
   task-migration gRPC use `InsecureServerCredentials` and
   `InsecureChannelCredentials` with no TLS
   (`src/rpc/task_migration.cpp`, `src/node_manager/spawn.cpp`,
   `src/task_manager/task_manager.cpp`), so query plans, result batches, and
   PGAS base pointers travel in plaintext. An observer on the path could read or
   inject them.

## Critical Security Assumptions

GQE relies on the deployment environment for the protections it does not implement
itself:

- **Trusted network.** The Flight SQL endpoint and internal gRPC ports are assumed
  to be reachable only from trusted hosts (e.g. bound to loopback or a private
  cluster network). There is no transport encryption or authentication; the
  `--address` bind is operator-controlled.
- **Trusted clients.** Any client that can connect is assumed to be authorized for
  full access — arbitrary SQL, DDL, and directing the server to read server-side
  filesystem paths. There is no per-client authorization or path sandboxing.
- **Trusted input data.** Parquet files and Substrait/physical plans are assumed to
  be well-formed and from a trusted source; the parsers are not hardened against
  adversarial input.
- **Trusted operator and host.** The operator supplies a trusted
  `--task-manager-binary`; spawned task-manager subprocesses inherit the node
  manager's environment. The host filesystem the server can access defines the
  confidentiality boundary for data.
- **Single trusted node/fabric for multi-GPU.** NVSHMEM and inter-process gRPC are
  assumed to run within one trusted node or cluster interconnect.
- **Proof-of-concept posture.** GQE is a research blueprint, not a multi-tenant or
  internet-facing service. Production use would require adding authentication,
  authorization, transport security, input hardening, and resource quotas.
- **Trusted, root-capable container runtime.** The provided container image runs
  as root with no `USER` directive by design; it is a proof-of-concept dev/build
  image, not a hardened multi-tenant or privileged deployment. Container-to-host
  isolation — the blast radius of root inside the container, host bind mounts,
  and device access — is assumed to be enforced by the deployment environment,
  not the image.
- **Trusted transport for internal traffic.** Internal gRPC has no transport
  encryption; confidentiality and integrity of node↔task and task-migration
  traffic are delegated to the trusted single node (loopback) or trusted cluster
  interconnect, per the assumptions above.
