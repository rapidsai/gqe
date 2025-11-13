# Contributing to GQE

Contributions to GQE fall into the following categories:

1. To report a bug, request a new feature, or report a problem with documentation, please file an
   [issue](placeholder_url) describing the problem or new feature in detail. The GQE team
   evaluates and triages issues, and schedules them for a release. If you believe the issue needs
   priority attention, please comment on the issue to notify the team.
2. To propose and implement a new feature, please file a new feature request
   [issue](placeholder_url). Describe the intended feature and
   discuss the design and implementation with the team and community. Once the team agrees that the
   plan looks good, go ahead and implement it, using the [code contributions](#code-contributions)
   guide below.
3. To implement a feature or bug fix for an existing issue, please follow the [code
   contributions](#code-contributions) guide below. If you need more context on a particular issue,
   please ask in a comment.

## Code contributions

### Your first issue

1. Follow the guide at the bottom of this page for
   [Setting up your build environment](#setting-up-your-build-environment).
2. Find an issue to work on. The best way is to look for the
   [good first issue](placeholder_url)
   or [help wanted](placeholder_url)
   labels.
3. Comment on the issue stating that you are going to work on it.
4. Create a fork of the GQE repository and check out a branch with a name that
   describes your planned work. For example, `fix-documentation`.
5. Write code to address the issue or implement the feature.
6. Add unit tests and unit benchmarks.
7. [Create your pull request](placeholder_url). To run continuous integration (CI) tests without requesting review, open a draft pull request.
8. Verify that CI passes all [status checks](placeholder_url).
   Fix if needed.
9. Wait for other developers to review your code and update code as needed.
   Changes require approval from GQE maintainers before merging.
10. Once reviewed and approved, a GQE developer will merge your pull request.

If you are unsure about anything, don't hesitate to comment on issues and ask for clarification!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you can look at the
prioritized issues for our next release in our
[project boards](placeholder_url).

Look at the unassigned issues, and find an issue to which you are comfortable contributing. Start
with _Step 3_ above, commenting on the issue to let others know you are working on it. If you have
any questions related to the implementation of the issue, ask them in the issue instead of the PR.

## Setting up your build environment
1. Begin by cloning the GQE repository in your workspace:
   ```bash
   git clone <GQE_REPO>
   ```
2. The easiest way to set up the GQE build environment is by running the [Docker image](placeholder_url) which has all dependencies installed. Follow the [Docker Engine install instructions](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) if you need to install Docker. If needed, [install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt) and [configure](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker) the NVIDIA container toolkit. With Docker set up, launch the GQE container:
   ```bash
   docker run -it --rm --gpus all \
            -v gqe:</gqe> \
            -v <HOST_DIR>:<CONTAINER_DIR> \
            <GQE_DOCKER_IMAGE>
   ```
3. Build GQE inside the container:
   ```bash
   mkdir /gqe/build
   cd $_
   cmake ..
   make -j$(nproc)
   ```
4. Run test cases to make sure setup is successsful:
   ```bash
   ctest --output-on-failure
   ```

## Developer Guidelines
For high-level design issues like interfaces, class hierarchies, recourse management, error handling etc., we will follow [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines). Given that GQE uses libcudf extensively, we will also follow [libcudf's developer guide](https://github.com/rapidsai/cudf/blob/main/cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md) where relevant.

### Naming convention
We generally use [snake_case](https://en.wikipedia.org/wiki/Snake_case) in accordance with [libcudf convention](https://github.com/rapidsai/cudf/blob/main/cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md#code-and-documentation-style-and-formatting). The main exception is in query compiler code, where [camelCase](https://en.wikipedia.org/wiki/Camel_case) is desirable for readability particularly when mixing with MLIR library code and/or using TableGen. See [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html#name-types-functions-variables-and-enumerators-properly) for more details.

### Code formatting
GQE uses clang-format with the same formatting style as used by libcudf. clang-format will be checked but not automatically applied by the CI. Developers are expected to apply clang-format locally before the MRs can be accepted. For any modified source file, one can run the following to apply formatting:
```bash
clang-format -i <source-file>
```
For convenience, it is recommended to configure your text editor to automatically apply clang-format on save. For example, vscode has a [plugin](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format) to apply clang-format automatically.

## Signing Your Work

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```
