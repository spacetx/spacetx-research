#!/usr/bin/env bash

# It does:
# 1. copy input.json to a gs://google_bucket/random_hash.json
# 2. create the tmp.json with the location of gs://google_bucket/random.json to run the wdl workflow
# 3. run cromshel, i.e.:
#    cromshell submit jupyter.wdl tmp.json
# 4. copy the input.json into the cromwell-v47.dsde-methods.broadinstitute.org/fd0c78d0-fc67-4c59-af79-e56e0c29b232 together with the wdl file 


# Define some defaults
BUCKET=${BUCKET:-"gs://ld-results-bucket/input_jsons"}
WDL=${WDL:-"jupyter.wdl"}
JSON=${JSON:-"parameters.json"}
SCRIPTNAME=$( echo $0 | sed 's#.*/##g' )
HERE=${PWD}

# Helper functions
display_help() {
  echo -e ""
  echo -e "-- $SCRIPTNAME --"
  echo -e ""
  echo -e " Submit wdl workflow using cromshell."
  echo -e " IMPORTANT: It assumes that the workflow has a SINGLE JSON FILE as INPUT"
  echo -e " The input json file is copied into the cloud and the path_to_json_in_the_cloud is passed to the workflow"
  echo -e " If the input json:"
  echo -e " 1. has the \"wdl.alias\" key then the cromshell alias comand is run"
  echo -e " 2. has the \"wdl.bucket_output\" key then the options.json file is generated and passed to cromshell"
  echo -e "    This will make sure that the output of the run are stored both in execution bucket and specified bucket_output"
  echo -e ""
  echo -e " Example usage:"
  echo -e "   $SCRIPTNAME $WDL $JSON $BUCKET"
  echo -e "   $SCRIPTNAME $WDL $JSON"
  echo -e "   $SCRIPTNAME $WDL"
  echo -e "   $SCRIPTNAME"
  echo -e "   $SCRIPTNAME -h"
  echo -e ""
  echo -e ""
  echo -e " Supported Flags:"
  echo -e "   -h or --help     Display this message"
  echo -e ""
  echo -e " Default behavior:"
  echo -e "   If no inputs are specified the default values will be used:"
  echo -e "   wdl_file --------> $WDL"
  echo -e "   json_file -------> $JSON"
  echo -e "   bucket_for_json -> $BUCKET"
  echo -e ""
  echo -e ""
}

exit_with_error() {
  echo -e ""
  echo -e "ERROR!. Something went wrong"
  exit 1
}

exit_with_success() {
  echo -e ""
  echo -e "GREAT!. Everything went smoothly"
  exit 0
}


#--------------------------------------
# 1. read inputs from command line
#--------------------------------------
POSITIONAL=()
while [[ $# -gt 0 ]]
do
KEY="$1"

case $KEY in
    -h|--help)
    display_help
    exit
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

for i in "${POSITIONAL[@]}"
do
   if [[ $i == *.json ]]; then JSON=$i; fi
   if [[ $i == *.wdl ]]; then WDL=$i; fi
   if [[ $i == gs://* ]]; then BUCKET=$i; fi
done

# At this point I have these trhee values:
echo "Current values: --" $WDL $JSON $BUCKET



#-----------------------------------------------------
# 2. make copy of input file to cloud with random hash
#-----------------------------------------------------
echo
echo "Step1: copying json file into google bucket"
RANDOM_HASH=$(cat /dev/urandom | env LC_CTYPE=C tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
PATH_TO_BUCKET="$BUCKET/${RANDOM_HASH}.json"
gsutil cp $JSON $PATH_TO_BUCKET
# check if file exist in bucket
gsutil -q stat $PATH_TO_BUCKET
if [[ $? != *0* ]]; then
   exit_with_error
fi



#-----------------------------------------------------------
# 3. create the tmp.json with the path_to_json_in_the_cloud 
#-----------------------------------------------------------
echo
echo "Step2: crerating tmp.json with the inputs for the workflow"
echo "tmp_params.json is:"
womtool inputs $WDL | jq --arg path $PATH_TO_BUCKET '[to_entries[] | {"key": .key, "value": $path }] | from_entries' | tee tmp_params.json

OUTPUT_BUCKET=$(cat $JSON | jq '.["wdl.bucket_output"]' | sed 's/"//g')
if [ $? == 0 ]; then
   # I have sucesfully found the desired key
   use_option_file="yes"
   echo "tmp_options.json is:"			       
   jq -n --arg output $OUTPUT_BUCKET '{"final_workflow_outputs_dir" : $output, 
        			       "use_relative_output_paths" : "false",
				       "final_workflow_log_dir": $output,
				       "final_call_logs_dir": $output}' | tee tmp_options.json
else 
   use_option_file="no"
fi

#-----------------------------------------------------
# 4. run cromshell
#-----------------------------------------------------
echo
echo "Step3: run cromshell"
if [ "$use_option_file" = "yes" ]; then
   echo "RUNNING: cromshell submit $WDL tmp_params.json tmp_options.json" 
   cromshell submit $WDL tmp_params.json tmp_options.json | tee tmp_run_status
   rm tmp_params.json tmp_options.json 
else
   echo "RUNNING: cromshell submit $WDL tmp_params.json" 
   cromshell submit $WDL tmp_params.json | tee tmp_run_status
   rm tmp_params.json
fi

# check I find the word Submitted in the cromshell_output
read -r ID_1 STATUS <<<$(tail -1 tmp_run_status | jq '.id, .status' | sed 's/\"//g')
rm tmp_run_status
if [ "$STATUS" != "Submitted" ]; then
   exit_with_error
fi

# run the alia command if necessary
ALIAS_NAME=$(cat $JSON | jq '.["wdl.alias"]' | sed 's/"//g')
if [ $? == 0 ]; then
   # Found alias key 
   cromshell 'alias' -1 $ALIAS_NAME
fi

#--------------------------------------------------------------
# 5. copy the parameter file used into the cromshell directory
#--------------------------------------------------------------
echo
echo "Step4: copy all json files into cromshell directory"
# read last line from the database.tsv to extract ID and OUTPUT_DIR
CROMSHELL_CONFIG_DIR=$HOME/.cromshell
read -r HTTPS_CROMWELL ID_2 <<<$( tail -1 "$CROMSHELL_CONFIG_DIR/all.workflow.database.tsv"  | awk '{print $2,$3}')
GENERAL_OUTPUT_DIR=$(echo $HTTPS_CROMWELL | sed 's/https:\/\///')  # remove string "https://" from HTTPS_CROMWELL
OUTPUT_DIR="$CROMSHELL_CONFIG_DIR/$GENERAL_OUTPUT_DIR/$ID_2"

echo $ID_1 $ID_2
if [ "$ID_1" = "$ID_2" ]; then
   # This means that last line of database correspond to run I just submitted
   if [ -d $OUTPUT_DIR ]; then
      # This means that the directory exists
      cp $JSON $OUTPUT_DIR  # the WDL file together with tmp_options.json and tmp_params.json are copied by cromshell automatically
      cd $OUTPUT_DIR
      echo "In directory $(pwd) there are the files:"
      ls -l
      cd $HERE
   fi	
else
   exit_with_error
fi

exit_with_success
