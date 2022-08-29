podTemplate(yaml: '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: ubuntu
    image: ubuntu:20.04
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
    container('ubuntu') {
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
            apt-get update -y && apt-get install -y --no-install-recommends wget ca-certificates
            wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh
            sh /miniconda.sh -b -p /conda
          '''
        }
        stage("Install clang-tools using conda") {
          sh '''#!/bin/bash
            /conda/bin/conda create --name clang-tools -y
            source /conda/bin/activate clang-tools
            conda install -q -c rapidsai -c nvidia -c conda-forge -y clang-tools=11.1.0
          '''
        }

        try {
          stage("Check clang-format") {
            sh '''#!/bin/bash
              source /conda/bin/activate clang-tools
              find ./include ./src -name *.hpp -o -name *.cpp -o -name *.cuh -o -name *.cu | xargs clang-format -style=file --dry-run -Werror
            '''
          }
          // FIXME: Add another stage for clang-tidy
          updateGitlabCommitStatus name: 'code style', state: 'success'
        } catch (exc) {
          updateGitlabCommitStatus name: 'code style', state: 'failed'
        }

      } catch (exc) {
        updateGitlabCommitStatus name: 'code style', state: 'failed'
        throw exc
      }
    }
  }
}
