#!/bin/bash

cat wav.scp | while read line;do
    utt=`echo $line | cut -d' ' -f1`
    wav=`echo $line | cut -d' ' -f2`
    dur=`soxi -d $wav | awk -F":" '{print $3}'`
    echo $utt" "$dur
done