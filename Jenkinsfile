podTemplate(yaml: '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: cuda
    image: nvcr.io/nvidia/cuda:11.5.2-devel-ubuntu20.04
    command:
    - cat
    resources:
          requests:
             memory: 4Gi
             cpu: 1
          limits:
             memory: 4Gi
             cpu: 1
    restartPolicy: Never
    backoffLimit: 4
    tty: true
  nodeSelector:
    kubernetes.io/os: linux
''') {
  node(POD_LABEL) {
    container('cuda') {
      updateGitlabCommitStatus name: "code style", state: "running"

      try {
        branch = env.gitlabMergeRequestLastCommit ?: "*/main"
        url = env.gitlabSourceRepoHttpUrl ?: "https://gitlab-master.nvidia.com/GPUDB/gqe.git"
        stage("Checkout code") {
          checkout([$class: 'GitSCM',
                    branches: [[name: branch]],
                    userRemoteConfigs: [[credentialsId: 'dtcomp-ci-pw',
                                         url: url]]])
        }
        stage("Install miniconda") {
          sh '''#!/bin/bash
            apt-get update -y && apt-get install -y --no-install-recommends wget ca-certificates build-essential
            wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh
            sh /miniconda.sh -b -p /conda
            /conda/bin/conda install -q -c conda-forge -y mamba -n base
          '''
        }
        stage("Install dependencies using conda") {
          sh '''#!/bin/bash
            /conda/bin/mamba env create -q -f conda/environment.yml
          '''
        }
        stage("Check style") {
          error = false

          def tests = [:]
          tests["clang-format"] = {
            try {
              sh '''#!/bin/bash
                source /conda/bin/activate gqe
                find ./include ./src -name *.hpp -o -name *.cpp -o -name *.cuh -o -name *.cu | xargs clang-format -style=file --dry-run -Werror
              '''
            } catch (exc) {
              error = true
            }
          }
          tests["clang-tidy"] = {
            try {
              sh'''#!/bin/bash
                source /conda/bin/activate gqe
                mkdir build && cd build && cmake .. && cd ..
                find ./include ./src -name *.hpp -o -name *.cpp -o -name *.cuh -o -name *.cu | xargs clang-tidy -p build --header-filter=.* --warnings-as-errors=*
              '''
            } catch (exc) {
              error = true
            }
          }
          parallel(tests)

          if (error) {
            updateGitlabCommitStatus name: 'code style', state: 'failed'
          } else {
            updateGitlabCommitStatus name: 'code style', state: 'success'
          }
        }
      } catch (exc) {
        updateGitlabCommitStatus name: 'code style', state: 'failed'
        throw exc
      }
    }
  }
}
