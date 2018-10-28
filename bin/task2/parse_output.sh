#!/bin/bash
input=$1
output=$2
cat $input | grep -w "CV" | grep "total" | awk {'print $2,$3,$4'} >> $output
