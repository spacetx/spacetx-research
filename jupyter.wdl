version 1.0

# Note that cromshell will automatically localize file in the following way:
# gs://ld-data-bucket/data/fashionmnist_test.pkl -> /cromwell_roo/data/fashionmnist_test.pkl

task parse_json {
    input {
        File input_json
        String prefix
    }

    parameter_meta {
        input_json: { localization_optional: false }  # this means file will be localized
    }

    command <<<

        python <<CODE

        import json

        # read the input json and write it back to have a copy
        with open("~{input_json}") as fp:
            my_dict = json.load(fp)

        # parse the dictionary for elements starting with the pattern and save it to a MAP output
        new_dict = {}
        for k,v in my_dict.items():
            if k.startswith("~{prefix}"):
                new_dict[k]=v
        print(json.dumps(new_dict))
        CODE
    >>>

    runtime {
        docker: "python"
        cpu: 1
        preemptible: 3
    }

    output {
        Map[String, String] output_map = read_json(stdout())
        File std_out = stdout()
    }
}

task run_jupyter_localize {
    input {
        File input_json

        File file_train
        File file_test
        File file_ckpt

        String dir_output
        String bucket_output

        String notebook_name
        String git_repo
        String commit_or_branch
    }

    parameter_meta {
        # the following means that all files will be localized
        file_train: { localization_optional: false }
        file_test: { localization_optional: false }
        file_ckpt: { localization_optional: false }
        input_json: { localization_optional: false }
    }


    command <<<

        #-----------------------------------------------#
        # 1. checkout the repo and the commit you want in the CHECKOUT_DIR
        #-----------------------------------------------#
        set -e
        git clone ~{git_repo} ./checkout_dir
        cd checkout_dir
        git checkout ~{commit_or_branch}
        hash=$(git log | head -1 | awk '{print $2}')
        echo "commit_or_branch " $hash

        #-----------------------------------------------#
        # 2. The files are localized in the EXECUTION_DIR while the code is in CHECKOUT_DIR
        # Move everything in the execution directory
        #-----------------------------------------------#
        cp -r ./* ../
        cd ..
        rm -rf ./*.wdl ./*.json
        echo $(ls)

        #-----------------------------------------------#
        # 3. replace the commit hash in the input_json file and rename it to parameters.json
        # This is what the notebook is expecting
        # Do this with python
        #-----------------------------------------------#
        python <<CODE
        import json
        with open("~{input_json}") as fp:
            my_dict = json.load(fp)
        my_dict["wdl.commit_or_branch"] = "$hash"
        with open("parameters.json", "w") as fp:
            json.dump(my_dict, fp)
        CODE

        #-----------------------------------------------#
        # 4. make a link from localized files to the execution_dir.
        # So that the notebook has all the files it needs in the execution dir
        #-----------------------------------------------#
        name_train=$(basename ~{file_train})
        name_test=$(basename ~{file_test})
        name_ckpt=$(basename ~{file_ckpt})
        ln -s ~{file_train} $name_train
        ln -s ~{file_test} $name_test
        ln -s ~{file_ckpt} $name_ckpt

        #-----------------------------------------------#
        # 5. prepare the run the notebook
        #-----------------------------------------------#
        echo "just before running notebook this is what I see in the execution directory:"
        echo $(ls)

        # REAL RUN
        #pip install matplotlib
        #pip install moviepy
        #pip install jupyter_contrib_nbextensions
        jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to=html --execute ~{notebook_name} --output notebook.html

        # FAKE RUN
        #mkdir -p ~{dir_output}
        #touch notebook.html
        #touch ~{dir_output}/trial.png

        # CHECK
        echo "in the output_directory there are:"
        echo $(ls ~{dir_output})

    >>>
    
    runtime {
        # USE THIS ONE FOR THE REAL RUN
        docker: "us.gcr.io/broad-dsde-methods/pyro_matplotlib:1.3.0"
        bootDiskSizeGb: 50
        memory: "15G"
        cpu: 4
        zones: "us-east1-d us-east1-c"
        gpuCount: 1
        gpuType: "nvidia-tesla-k80" # "nvidia-tesla-p100" 
        maxRetries: 0
    }

###    runtime {
###        # USE THIS ONE FOR THE FAKE RUN 
###        docker: "python"
###        cpu: 1
###        preemptible: 3
###    }

    output {
        File output_html = "notebook.html"
        File params = "parameters.json"
        Array[File] results = glob("~{dir_output}/*")
    }
}




workflow jupyter_localize {

    input {
        File parameters_json
    }

    call parse_json {
        input :
            input_json = parameters_json,
            prefix = "wdl."
    }

    call run_jupyter_localize {
        input :
            input_json = parameters_json,
            file_train = parse_json.output_map["wdl.file_train"],
            file_test = parse_json.output_map["wdl.file_test"],
            file_ckpt = parse_json.output_map["wdl.file_ckpt"],
            bucket_output = parse_json.output_map["wdl.bucket_output"],
            dir_output = parse_json.output_map["wdl.dir_output"],
            notebook_name = parse_json.output_map["wdl.notebook_name"],
            git_repo = parse_json.output_map["wdl.git_repo"],
            commit_or_branch = parse_json.output_map["wdl.commit_or_branch"]
    }

}






