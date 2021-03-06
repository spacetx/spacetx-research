version 1.0

# Note that cromshell will automatically localize file in the following way:
# gs://ld-data-bucket/data/fashionmnist_test.pkl -> /cromwell_roo/data/fashionmnist_test.pkl

task train {
    input {
        File ML_parameters
        File data_train
        File data_test
        File ckpt
        File ground_truth 
        File credentials_json
        String git_repo
        String git_branch_or_commit
        String main_file_name 
    }


    command <<<
        exec_dir=$(pwd)
        echo "--> $exec_dir"
        echo "START --> Content of exectution dir"
        echo $(ls)
        
        # 1. checkout the repo in the checkout_dir
        set -e
        git clone ~{git_repo} ./checkout_dir
        cd ./checkout_dir
        git checkout ~{git_branch_or_commit}
        echo "AFTER GIT --> Content of checkout dir"
        echo $(ls)
        
        # 2. link the file which have been localized to checkout_dir/src
        # and give them the name the main.py expects
        ln -s ~{ML_parameters} ./src/ML_parameters.json
        ln -s ~{data_train} ./src/data_train.pt
        ln -s ~{data_test} ./src/data_test.pt
        ln -s ~{ckpt} ./src/ckpt.pt
        ln -s ~{ground_truth} ./src/ground_truth
        echo "AFTER CHANGING NAMES --> Content of checkout dir"
        echo $(ls)

        # 3. run python code only if NEPTUNE credentials are found
        # extract neptune_token from json file using regexpression
        token=$(cat ~{credentials_json} | grep -o '"NEPTUNE_API_TOKEN"\s*:\s*"[^"]*"' | grep -o '"[^"]*"$')
        if [ ! -z $token ]; then
           export NEPTUNE_API_TOKEN=$token
           cd ./src 
           python ~{main_file_name}
        fi
    >>>
    
#    runtime {
#          docker: "python"
#          cpu: 1
#          preemptible: 3
#    }
    
    runtime {
         docker: "us.gcr.io/broad-dsde-methods/pyro_matplotlib:0.0.7"
         bootDiskSizeGb: 100
         memory: "26G"
         cpu: 4
         zones: "us-east1-d us-east1-c"
         gpuCount: 1
         gpuType:  "nvidia-tesla-p100" #"nvidia-tesla-k80"
         maxRetries: 0
         preemptible_tries: 0
    }

}

workflow neptune_ml {

    input {
        File ML_parameters 
        File data_train 
        File data_test
        File ckpt
        File ground_truth 
        File credentials_json 
        String git_repo
        String git_branch_or_commit 
        String main_file_name 
    }

    call train { 
        input :
            ML_parameters = ML_parameters,
            credentials_json = credentials_json,
            data_train = data_train,
            data_test = data_test,
            ckpt = ckpt,
            ground_truth = ground_truth, 
            git_repo = git_repo,
            git_branch_or_commit = git_branch_or_commit,
            main_file_name = main_file_name 
    }
}
