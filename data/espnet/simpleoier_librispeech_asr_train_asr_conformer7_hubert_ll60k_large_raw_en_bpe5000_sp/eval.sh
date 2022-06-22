#!/bin/bash
# Copyright 2022 Amazon.com, Inc. and its affiliates. All Rights Reserved.
#  
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# 
# Licensed under the Amazon Software License (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
# 
#   http://aws.amazon.com/asl/
# 
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Use sclite to compute WER
# Make sure you've install SCTK https://github.com/usnistgov/SCTK
sclite=  # Set your sclite tool path

if [ -z "$sclite" ]; then
  echo \$sclite is not set.
  echo Make sure you\'ve install SCTK https://github.com/usnistgov/SCTK
  echo and set the sclite tool path in $0
  exit 1
fi

if [ $# -ne 3 ]; then
  echo "USAGE: $0 <hyp_dir>  <ref_dir>  <output_dir>"
  echo "   hyp_dir:  hypothesis folder. The folder should contains 3 files"
  echo "             hyp_wer.trn / hyp_cer.trn / hyp_ter.trn"
  echo "             for computing WER, CER, and TER"
  echo "   ref_dir:  reference folder. The folder should contains 3 files"
  echo "             ref_wer.trn / ref_cer.trn / ref_ter.trn"
  echo "             for computing WER, CER, and TER"
  echo "   output_dir:  folder to output result_wer.txt, result_cer.txt, and result_ter.txt"
  exit 1
fi
hyp_dir=$1
ref_dir=$2
output_dir=$3

for type in wer cer ter; do
    tokenized_hyp=$hyp_dir/hyp_${type}.trn
    if [ ! -e $tokenized_hyp ]; then
      echo "ERROR: Cannot fine $tokenized_hyp"
      exit 1
    fi
    ${sclite} -r "${ref_dir}/ref_${type}.trn" trn \
            -h "${tokenized_hyp}" trn \
            -i rm -o all stdout > "${output_dir}result_${type}.txt"
done


for type in wer cer ter; do
    echo $type
    grep -e Avg -e SPKR -m 2 "${output_dir}result_${type}.txt"
done
