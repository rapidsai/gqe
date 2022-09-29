def checkout_code() {
  branch = env.gitlabMergeRequestLastCommit ?: "*/main"
  url = env.gitlabSourceRepoHttpUrl ?: "https://gitlab-master.nvidia.com/GPUDB/gqe.git"
  stage("Checkout code") {
    checkout(
      [
        $class: 'GitSCM',
        branches: [[name: branch]],
        doGenerateSubmoduleConfigurations: false,
        extensions: [
          [
            $class: 'SubmoduleOption',
            disableSubmodules: false,
            parentCredentials: true,
            recursiveSubmodules: false,
            reference: '',
            trackingSubmodules: false
          ]
        ],
        submoduleCfg: [],
        userRemoteConfigs: [
          [
            credentialsId: 'dtcomp-ci-pw',
            url: url
          ]
        ]
      ]
    )
  }
}

def install_dependencies_with_conda() {
  stage("Install miniconda") {
    sh '''#!/bin/bash
      apt-get update -y && apt-get install -y --no-install-recommends wget git ca-certificates build-essential
      wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O /miniconda.sh
      sh /miniconda.sh -b -p /conda
      /conda/bin/conda install -q -c conda-forge -y mamba -n base
    '''
  }
  stage("Install dependencies using conda") {
    sh '''#!/bin/bash
      /conda/bin/mamba env create -q -f conda/environment.yml
    '''
  }
}

def tests = [:]

tests["code style"] = {
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
        checkout_code()
        install_dependencies_with_conda()

        try {
          stage("clang-format") {
            sh '''#!/bin/bash
              source /conda/bin/activate gqe
              find ./include ./src -name *.hpp -o -name *.cpp -o -name *.cuh -o -name *.cu | xargs clang-format -style=file --dry-run -Werror
            '''
          }
          stage("clang-tidy") {
            sh'''#!/bin/bash
              source /conda/bin/activate gqe
              mkdir build && cd build && cmake -DProtobuf_USE_STATIC_LIBS=ON .. && cd ..
              find ./include ./src -name *.hpp -o -name *.cpp -o -name *.cuh -o -name *.cu | xargs clang-tidy -p build --warnings-as-errors=*
            '''
          }
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
}  // code style

tests["CUDA 11.5 conda"] = {
podTemplate (yaml: '''
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
             nvidia.com/gpu: 1
          limits:
             nvidia.com/gpu: 1
    restartPolicy: Never
    backoffLimit: 4
    tty: true
  nodeSelector:
    kubernetes.io/os: linux
    nvidia.com/gpu_type: "TITAN_V"
    nvidia.com/driver_version: 510.41
''') {
  node(POD_LABEL) {
    container('cuda') {
      updateGitlabCommitStatus name: "CUDA 11.5 conda", state: "running"

      try {
        checkout_code()
        install_dependencies_with_conda()

        try {
          stage("Run tests") {
            sh'''#!/bin/bash
              hostname
              nvidia-smi
              source /conda/bin/activate gqe
              mkdir build && cd build && cmake -DProtobuf_USE_STATIC_LIBS=ON .. && make -j8 && ctest --output-on-failure
            '''
          }
          updateGitlabCommitStatus name: "CUDA 11.5 conda", state: "success"
        } catch (exc) {
          updateGitlabCommitStatus name: "CUDA 11.5 conda", state: "failed"
        }
      } catch (exc) {
        updateGitlabCommitStatus name: "CUDA 11.5 conda", state: "failed"
        throw exc
      }
    }
  }
}
}

parallel(tests)
