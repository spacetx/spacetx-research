#!/usr/bin/env bash

# Define some defaults
DEFAULT_BUCKET="gs://ld-tmp-storage/input_jsons"
DEFAULT_WDL="neptune_ml.wdl"
DEFAULT_WDL_JSON="WDL_parameters.json"
DEFAULT_ML_JSON="ML_parameters.json"

# Set variables to default
BUCKET=$DEFAULT_BUCKET
WDL=$DEFAULT_WDL
WDL_JSON=$DEFAULT_WDL_JSON
ML_JSON=$DEFAULT_ML_JSON
HERE=${PWD}
SCRIPTNAME=$( echo $0 | sed 's#.*/##g' )

# Helper functions
display_help() {
  echo -e ""
  echo -e "-- $SCRIPTNAME --"
  echo -e ""
  echo -e " Submit wdl workflow using cromshell."
  echo -e ""
  echo -e " Example usage:"
  echo -e "   $SCRIPTNAME $WDL $WDL_JSON --ml $ML_JSON -b $BUCKET"
  echo -e "   $SCRIPTNAME -h"
  echo -e ""
  echo -e " Supported Flags:"
  echo -e "   -h or --help     Display this message"
  echo -e "   -m or --ml       Name of json file with all the parameters of the ML model. This file will be provided as-is to pytorch code"
  echo -e "   -b or --bucket   Name of google bucket where local files will be copied (VM will then localize those files)"
  echo -e "   -t or --template Show the template for $WDL_JSON" 
  echo -e ""
  echo -e " Default behavior (can be changed manually by editing the $SCRIPTNAME):"
  echo -e "   If no inputs are specified the default values will be used:"
  echo -e "   wdl_file -----------> $WDL"
  echo -e "   wdl_json_file ------> $WDL_JSON"
  echo -e "   ml_json_file -------> $ML_JSON"
  echo -e "   bucket -------------> $BUCKET"
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

template_wdl_json() {
  echo -e ""
  echo -e "Based on $WDL the template for $WDL_JSON is:"
  womtool inputs $WDL | sed '/ML_parameters/d'
}

#--------------------------------------
# 1. read inputs from command line
#--------------------------------------
POSITIONAL=""
while [[ $# -gt 0 ]]; do
	case "$1" in
		-h|--help)
			display_help
			exit 0
			;;
		-m|--ml)
			ML_JSON=$2
			shift
			;;
		-b|--bucket)
			BUCKET=$2
			shift
			;;
		-t|--template)
			template_wdl_json
			exit 0
			;;
		-*|--*=) # unknown option
			echo "ERROR: Unsupported flag $1"
			exit 1
			;;
		*) # positional
			if [[ $1 == *.wdl ]]; then WDL=$1; fi
			if [[ $1 == *.json ]]; then WDL_JSON=$1; fi
			shift 
			;;
	esac
done  # end of while loop

# At this point I have these trhee values:
echo "Current values: -->" $WDL $WDL_JSON $ML_JSON $BUCKET

# 1. copy ML_JSON in the cloud with random hash
echo
echo "Step1: copying $ML_JSON  into google bucket"
RANDOM_HASH=$(cat /dev/urandom | env LC_CTYPE=C tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
ML_JSON_CLOUD="$BUCKET/ml_${RANDOM_HASH}.json"
gsutil cp $ML_JSON $ML_JSON_CLOUD 

# 2. create the json file which will be passed to cromshell
echo
echo "Step2: crerating input.json file for cromshell"
key_for_ML_parameters=$(womtool inputs neptune_ml.wdl | jq 'keys[]' | grep "ML_par")
echo '{' "$key_for_ML_parameters" : '"'"$ML_JSON_CLOUD"'" }' > tmp.json
jq -s '.[0] * .[1]' tmp.json $WDL_JSON | tee input.json
rm -rf tmp.json

# 3. run cromshell
echo
echo "Step3: run cromshell"
echo "RUNNING: cromshell submit $WDL input.json"
cromshell submit $WDL input.json | tee tmp_run_status
rm -rf input.json 

# 4. check I find the word Submitted in the cromshell_output
echo
echo "Step4: check submission"
read -r ID_1 STATUS <<<$(tail -1 tmp_run_status | jq '.id, .status' | sed 's/\"//g')
rm tmp_run_status
if [ "$STATUS" != "Submitted" ]; then
   exit_with_error
fi

exit_with_success
